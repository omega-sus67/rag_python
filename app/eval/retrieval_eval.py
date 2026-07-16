"""
Retrieval-quality evaluation harness.

Measures whether the custom hierarchical + sliding-window semantic chunking
pipeline actually retrieves better chunks than a naive fixed-size token
splitter, using the same embedding model and the same pgvector search for
both. Metrics:

- hit-rate@k: fraction of queries where at least one of the top-k retrieved
  chunks (a) comes from the document the question is about AND (b) contains
  one of the expected answer substrings.
- MRR (mean reciprocal rank): average of 1/rank of the first hit, 0 if no
  hit within the deepest k. Rewards putting the right chunk *first*, not
  just somewhere in the top k.

Both strategies are ingested under namespaced file IDs (eval_semantic_*,
eval_baseline_*) so they never collide with regular documents or each other.
"""
import os
import hashlib
from typing import List, Dict, Any, Sequence

from sqlalchemy import select, delete

from app.db.database_manager import DatabaseManager, dbFile, dbChunk
from app.engines.hierarchical_chunker import HierarchicalSemanticEngine
from app.engines.semantic_chunker import SemanticEngine
from app.utils.token_optimizer import TokenSizeOptimizer
from app.utils.pdf_extractor import parsePdf
from app.core.config import settings

STRATEGIES = ("semantic", "baseline")


class RetrievalEvaluator:
    """
    Ingests an evaluation corpus under two chunking strategies and scores
    retrieval quality for each against a question dataset.
    """

    def __init__(self, db_manager: DatabaseManager, vector_engine: SemanticEngine):
        self.db_manager = db_manager
        self.vector_engine = vector_engine
        # The real pipeline: AST parse -> sliding-window semantic chunking.
        self.semantic_engine = HierarchicalSemanticEngine(
            vector_engine=vector_engine,
            window_size=settings.window_size,
            threshold_factor=settings.threshold_factor,
        )
        # The control: fixed-size token windows with overlap, no semantics.
        self.baseline_splitter = TokenSizeOptimizer()

    @staticmethod
    def _file_id(text_hash: str, strategy: str) -> str:
        return f"eval_{strategy}_{text_hash}"

    async def prepare_corpus(self, corpus_dir: str, documents: Sequence[str], force: bool = False) -> Dict[str, Any]:
        """
        Parses each PDF once, then chunks + embeds + stores it under both
        strategies. Skips documents already ingested unless force=True.
        Returns per-strategy chunk statistics.
        """
        stats: Dict[str, Any] = {s: {"documents": 0, "chunks": 0, "skipped": 0} for s in STRATEGIES}

        for filename in documents:
            path = os.path.join(corpus_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Eval corpus document missing: {path}")

            doc = parsePdf(path)
            text_hash = hashlib.sha256(doc.extracted_text.encode("utf-8")).hexdigest()

            for strategy in STRATEGIES:
                file_id = self._file_id(text_hash, strategy)

                async with self.db_manager.SessionLocal() as session:
                    existing = (await session.execute(select(dbFile.id).where(dbFile.id == file_id))).scalar()
                    if existing and not force:
                        stats[strategy]["skipped"] += 1
                        continue
                    if existing:
                        await session.execute(delete(dbChunk).where(dbChunk.file_id == file_id))
                        await session.execute(delete(dbFile).where(dbFile.id == file_id))
                        await session.commit()

                if strategy == "semantic":
                    rendered = await self.semantic_engine.process_document(doc.extracted_text, source_name=filename)
                    payloads = [(rc.text, rc.embeddings) for rc in rendered]
                else:
                    texts = self.baseline_splitter.optimize_block(doc.extracted_text)
                    vectors = await self.vector_engine.get_embeddings_async(texts)
                    payloads = [(t, v.tolist()) for t, v in zip(texts, vectors)]

                async with self.db_manager.SessionLocal() as session:
                    session.add(dbFile(
                        id=file_id,
                        title=f"{filename} [{strategy}]",
                        extracted_text=doc.extracted_text,
                    ))
                    # Flush the parent row first: without an ORM relationship
                    # between dbFile and dbChunk, the unit of work may emit the
                    # chunk INSERTs before the file INSERT and trip the FK.
                    await session.flush()
                    for index, (chunk_text, vector) in enumerate(payloads):
                        session.add(dbChunk(
                            id=f"{file_id}_{index}",
                            file_id=file_id,
                            chunk_index=index,
                            text_data=chunk_text,
                            embeddings=vector,
                        ))
                    await session.commit()

                stats[strategy]["documents"] += 1
                stats[strategy]["chunks"] += len(payloads)

        return stats

    async def _search(self, query_vector: List[float], strategy: str, top_k: int) -> List[Dict[str, Any]]:
        """Cosine-distance search restricted to one strategy's chunks."""
        distance = dbChunk.embeddings.cosine_distance(query_vector)
        statement = (
            select(dbChunk.text_data, dbFile.title, distance.label("distance"))
            .join(dbFile, dbChunk.file_id == dbFile.id)
            .where(dbFile.id.like(f"eval_{strategy}_%"))
            .order_by("distance")
            .limit(top_k)
        )
        async with self.db_manager.SessionLocal() as session:
            rows = (await session.execute(statement)).all()
        return [{"text": r[0], "title": r[1], "distance": float(r[2])} for r in rows]

    @staticmethod
    def _is_hit(row: Dict[str, Any], entry: Dict[str, Any]) -> bool:
        """
        A retrieved chunk is a hit only if it belongs to the document the
        question targets AND contains one of the expected answer substrings.
        The document constraint prevents false positives from generic words
        appearing in unrelated books.
        """
        if not row["title"].startswith(entry["document_title"]):
            return False
        text = row["text"].lower()
        return any(sub.lower() in text for sub in entry["expected_substrings"])

    async def evaluate(self, dataset: List[Dict[str, Any]], ks: Sequence[int] = (1, 3, 5)) -> Dict[str, Any]:
        """
        Runs every dataset query against both strategies and aggregates
        hit-rate@k for each k plus MRR (computed at the deepest k).
        """
        max_k = max(ks)
        per_strategy = {s: {"hits_at": {k: 0 for k in ks}, "rr_sum": 0.0, "details": []} for s in STRATEGIES}

        for entry in dataset:
            query_vector = (await self.vector_engine.get_embeddings_async([entry["query"]]))[0].tolist()

            for strategy in STRATEGIES:
                rows = await self._search(query_vector, strategy, max_k)
                first_hit_rank = next(
                    (rank for rank, row in enumerate(rows, start=1) if self._is_hit(row, entry)),
                    None,
                )

                bucket = per_strategy[strategy]
                if first_hit_rank is not None:
                    bucket["rr_sum"] += 1.0 / first_hit_rank
                    for k in ks:
                        if first_hit_rank <= k:
                            bucket["hits_at"][k] += 1

                bucket["details"].append({
                    "query": entry["query"],
                    "document": entry["document_title"],
                    "first_hit_rank": first_hit_rank,
                })

        n = len(dataset)
        results = {"num_queries": n, "ks": list(ks), "strategies": {}}
        for strategy, bucket in per_strategy.items():
            results["strategies"][strategy] = {
                "hit_rate_at": {k: bucket["hits_at"][k] / n for k in ks},
                "mrr": bucket["rr_sum"] / n,
                "details": bucket["details"],
            }
        return results

    async def cleanup_corpus(self) -> int:
        """Removes all eval documents and their chunks from the database."""
        async with self.db_manager.SessionLocal() as session:
            eval_ids = [
                r[0] for r in
                (await session.execute(select(dbFile.id).where(dbFile.id.like("eval_%")))).all()
            ]
            if eval_ids:
                await session.execute(delete(dbChunk).where(dbChunk.file_id.in_(eval_ids)))
                await session.execute(delete(dbFile).where(dbFile.id.in_(eval_ids)))
                await session.commit()
            return len(eval_ids)
