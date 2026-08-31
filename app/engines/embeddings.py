"""
Embedding providers.

The pipeline only ever needs one operation — turn a list of strings into a
matrix of unit-length vectors — so that is the entire seam between the chunker
and whatever actually computes embeddings.

Two implementations sit behind it:

  local   sentence-transformers in-process. What the eval numbers in the README
          were measured with, and what a GPU box should use. Pulls torch, which
          is ~2.5 GB of wheels and ~1-1.5 GB of resident memory.
  gemini  A hosted HTTP API. No torch, no model weights, ~80 MB of resident
          memory. This is what makes the worker fit in a free-tier container.

gemini-embedding-001 emits 3072 dimensions by default, so every request asks for
settings.embedding_dimension (768) via outputDimensionality — matching
bge-base-en-v1.5 and keeping the existing Vector(768) column and every stored
vector valid across the swap. Those truncated vectors are not unit length, so
both providers L2-normalize and cosine distance means the same thing either way.
"""

import time
from typing import List, Protocol

import httpx
import numpy as np

from app.core.config import settings


class EmbeddingProvider(Protocol):
    """Anything that can turn texts into a (len(texts), dim) float matrix."""

    def embed(self, texts: List[str]) -> np.ndarray: ...


def _clip(texts: List[str]) -> List[str]:
    """
    Clip inputs to the hosted providers' length ceiling.

    Local sentence-transformers silently truncate past the model's context
    (512 tokens for bge-base). Hosted APIs do not — Jina returns
    INPUT_TOKEN_LIMIT_EXCEEDED and fails the entire batch, which fails the whole
    document. One malformed input should not cost an ingestion.

    This matters in practice because the hierarchical chunker embeds *sentences*,
    and sentence splitting degrades badly on structured PDFs: a table in a
    standards document parses as one enormous pseudo-sentence. Caught exactly
    that on iso27001.pdf, which ingests locally and 400s against the API.
    """
    limit = settings.embedding_max_input_chars
    return [text if len(text) <= limit else text[:limit] for text in texts]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """
    Scale each row to unit length so a dot product is a cosine similarity.
    Zero-norm rows are left alone rather than turned into NaNs.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class LocalEmbeddingProvider:
    """
    sentence-transformers, loaded lazily and unloaded after inactivity.

    The import is deliberately inside the method rather than at module scope:
    the production image does not install torch at all, and importing this
    module must not fail there.
    """

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(settings.embedding_model)
        return self._model

    def warm(self):
        """Force the weights to load now instead of on the first encode."""
        self._load()

    def unload(self):
        """Drop the model and release CUDA cache, if torch is even present."""
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def embed(self, texts: List[str]) -> np.ndarray:
        return self._load().encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
        )


class GeminiEmbeddingProvider:
    """
    Google's batchEmbedContents endpoint.

    Two things this has to get right that a naive client would not:

    1. The API caps a batch at 100 items. A 300-chunk document is three calls,
       not one 300-item call that 400s.
    2. It is a network hop inside the ingestion path, so it retries on 429 and
       5xx with exponential backoff. Without this, one rate-limit response
       fails an entire document ingestion.
    """

    _MAX_ATTEMPTS = 5

    def __init__(self):
        self.api_key = settings.embedding_api_key or settings.llm_api_key
        self.model = settings.embedding_api_model
        self.api_base = settings.embedding_api_base.rstrip("/")
        self._MAX_BATCH = settings.embedding_api_batch_size
        # Set once a batch call 404s, so we stop retrying an endpoint this model
        # does not expose and use the per-item one for the rest of the run.
        self._batch_unsupported = False

    def _endpoint(self, method: str) -> str:
        return f"{self.api_base}/models/{self.model}:{method}?key={self.api_key}"

    def _content(self, text: str) -> dict:
        """
        One request body element.

        outputDimensionality is not optional for us: gemini-embedding-001 returns
        3072 dimensions by default, and the doc_chunks.embeddings column is
        Vector(768). Asking for 768 keeps every stored vector comparable with
        every new one. The truncated vectors are not unit length, which is why
        embed() re-normalizes afterwards.
        """
        return {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": settings.embedding_dimension,
        }

    def _post_single(self, client: httpx.Client, texts: List[str]) -> List[List[float]]:
        """Per-item fallback for models that expose embedContent but not the batch form."""
        out = []
        for text in texts:
            response = client.post(self._endpoint("embedContent"), json=self._content(text))
            if response.status_code != 200:
                raise RuntimeError(
                    f"Embedding API rejected embedContent ({response.status_code}): {response.text}"
                )
            out.append(response.json()["embedding"]["values"])
        return out

    def _post_batch(self, client: httpx.Client, batch: List[str]) -> List[List[float]]:
        if self._batch_unsupported:
            return self._post_single(client, batch)

        payload = {"requests": [self._content(text) for text in batch]}

        last_error = None
        for attempt in range(self._MAX_ATTEMPTS):
            try:
                response = client.post(self._endpoint("batchEmbedContents"), json=payload)
                if response.status_code == 200:
                    return [item["values"] for item in response.json()["embeddings"]]
                # Not every embedding model exposes the batch endpoint. Fall back
                # to per-item calls rather than failing the whole ingestion.
                if response.status_code == 404:
                    self._batch_unsupported = True
                    return self._post_single(client, batch)
                # 429 = rate limited, 5xx = transient upstream fault. Both are
                # worth retrying; a 400 is our bug and should surface now.
                if response.status_code != 429 and response.status_code < 500:
                    raise RuntimeError(
                        f"Embedding API rejected the request ({response.status_code}): {response.text}"
                    )
                last_error = RuntimeError(f"Embedding API {response.status_code}: {response.text}")
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                # Timeouts are transient on a shared free tier; retry them.
                last_error = exc

            if attempt < self._MAX_ATTEMPTS - 1:
                time.sleep(2**attempt)

        raise RuntimeError(f"Embedding API failed after {self._MAX_ATTEMPTS} attempts: {last_error}")

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self.api_key:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=gemini requires EMBEDDING_API_KEY (or LLM_API_KEY) to be set."
            )

        texts = _clip(texts)
        vectors: List[List[float]] = []
        with httpx.Client(timeout=settings.embedding_api_timeout) as client:
            for start in range(0, len(texts), self._MAX_BATCH):
                vectors.extend(self._post_batch(client, texts[start : start + self._MAX_BATCH]))

        # The API does not promise unit vectors; normalize so cosine distance
        # behaves identically to the local provider.
        return _l2_normalize(np.array(vectors, dtype=np.float32))


class OpenAICompatibleEmbeddingProvider:
    """
    Any embeddings API that speaks the OpenAI wire format:

        POST {base}/embeddings
        Authorization: Bearer <key>
        {"model": "...", "input": ["text", ...]}
        -> {"data": [{"embedding": [...], "index": 0}, ...]}

    That one shape covers Jina, Together, DeepInfra, Mistral, Nebius, and OpenAI
    itself, so switching vendors is an env-var change rather than a new class.
    Deliberately generic: after being locked out of one provider mid-deploy, the
    cost of moving to the next one should be near zero.

    Vendor is chosen entirely by EMBEDDING_API_BASE + EMBEDDING_API_MODEL.
    """

    _MAX_ATTEMPTS = 5

    def __init__(self):
        self.api_key = settings.embedding_api_key or settings.llm_api_key
        self.model = settings.embedding_api_model
        self.api_base = settings.embedding_api_base.rstrip("/")
        self._MAX_BATCH = settings.embedding_api_batch_size

    def _post_batch(self, client: httpx.Client, batch: List[str]) -> List[List[float]]:
        payload = {"model": self.model, "input": batch}
        # Only some models (OpenAI's text-embedding-3-*) accept a dimensions
        # override. Sending it to one that does not is a 400, so it stays opt-in.
        if settings.embedding_api_send_dimensions:
            payload["dimensions"] = settings.embedding_dimension

        last_error = None
        for attempt in range(self._MAX_ATTEMPTS):
            try:
                response = client.post(
                    f"{self.api_base}/embeddings",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                if response.status_code == 200:
                    data = response.json()["data"]
                    # The spec allows results out of order; index restores it.
                    data.sort(key=lambda item: item.get("index", 0))
                    return [item["embedding"] for item in data]
                if response.status_code != 429 and response.status_code < 500:
                    raise RuntimeError(
                        f"Embedding API rejected the request ({response.status_code}): {response.text}"
                    )
                last_error = RuntimeError(f"Embedding API {response.status_code}: {response.text}")
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                # Timeouts are transient on a shared free tier; retry them.
                last_error = exc

            if attempt < self._MAX_ATTEMPTS - 1:
                time.sleep(2**attempt)

        raise RuntimeError(f"Embedding API failed after {self._MAX_ATTEMPTS} attempts: {last_error}")

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self.api_key:
            raise RuntimeError(
                f"EMBEDDING_PROVIDER={settings.embedding_provider} requires EMBEDDING_API_KEY "
                "(or LLM_API_KEY) to be set."
            )

        texts = _clip(texts)
        vectors: List[List[float]] = []
        with httpx.Client(timeout=settings.embedding_api_timeout) as client:
            for start in range(0, len(texts), self._MAX_BATCH):
                vectors.extend(self._post_batch(client, texts[start : start + self._MAX_BATCH]))

        return _l2_normalize(np.array(vectors, dtype=np.float32))


def build_provider() -> EmbeddingProvider:
    """Selects the provider named by EMBEDDING_PROVIDER."""
    provider = settings.embedding_provider.lower()
    if provider in ("gemini", "google"):
        return GeminiEmbeddingProvider()
    if provider in ("local", "sentence-transformers", "st"):
        return LocalEmbeddingProvider()
    # Everything else is assumed to speak the OpenAI embeddings wire format.
    if provider in ("openai", "jina", "together", "deepinfra", "mistral", "api", "compatible"):
        return OpenAICompatibleEmbeddingProvider()
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER '{settings.embedding_provider}' "
        "(expected 'local', 'gemini', or an OpenAI-compatible name such as 'jina'/'openai')."
    )
