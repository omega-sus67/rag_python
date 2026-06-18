# RAG Pipeline — Engine Documentation

This document explains every class and method in `semantic_chunker.py`, `markdown_parser.py`, and `hierarchical_chunker.py`. For each method, it details how it changes the document, what information it provides for the RAG pipeline, and provides a line-by-line explanation of its operations on arguments and its return values.

---

## File: `semantic_chunker.py`

### Class: `SemanticEngine`

#### `__init__(self)`
**How it changes the document:** It does not change the document. 
**What information it provides for the RAG pipeline:** Provides the embedding model initialized for the pipeline to convert text into vector representations.
**Line-by-line explanation:** Line 9 initializes the `SentenceTransformer` model using the "all-MiniLM-L6-v2" architecture and stores it in the instance variable `self.model`. It returns `None`.

#### `get_embeddings(self, texts: List[str]) -> np.ndarray`
**How it changes the document:** It does not change the document.
**What information it provides for the RAG pipeline:** Provides a 2D array of vector embeddings for any given list of text strings, enabling semantic similarity comparisons.
**Line-by-line explanation:** Lines 12-13 check if the input `texts` list is empty, and if so, return an empty NumPy array of shape (0, 384). Lines 15-21 encode the `texts` into embeddings using the stored `self.model`, configuring it to convert outputs to NumPy arrays, show a progress bar, use a batch size of 32, and normalize the embeddings. Line 22 returns these computed `embeddings` as a NumPy `ndarray`.

#### `calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float`
**How it changes the document:** It does not change the document.
**What information it provides for the RAG pipeline:** Provides a numerical similarity score between two vector representations, used to detect semantic shifts.
**Line-by-line explanation:** Line 25 computes the dot product of `vec_a` and `vec_b`. Lines 26-27 compute the L2 norm for both vectors. Lines 29-30 check if either norm is 0.0 to prevent division by zero, returning 0.0 if true. Line 32 calculates and returns the cosine similarity by dividing the dot product by the product of the norms, returning the result as a `float`.

#### `calculate_similarity_matrix(embeddings: np.ndarray) -> np.ndarray`
**How it changes the document:** It does not change the document.
**What information it provides for the RAG pipeline:** Provides an NxN similarity matrix detailing the pairwise cosine similarities for an entire set of embeddings.
**Line-by-line explanation:** Line 36 calculates the L2 norms for each row in the `embeddings` array, preserving dimensions. Line 37 sets any zero norms to 1.0 to avoid division by zero errors. Line 38 normalizes the embeddings by dividing them by their respective norms. Line 40 computes the similarity matrix by calculating the dot product between the normalized embeddings and their transpose. Line 41 returns the calculated `similarity_matrix` as a NumPy array.

### Class: `SemanticBoundaryDetector`

#### `__init__(self, vector_engine: SemanticEngine, threshold_factor: float = 0.8)`
**How it changes the document:** It does not change the document.
**What information it provides for the RAG pipeline:** Initializes the boundary detector with a specific vector engine and sensitivity threshold for detecting topic shifts.
**Line-by-line explanation:** Lines 45-46 store the provided `vector_engine` and `threshold_factor` to `self`. Line 47 compiles a regular expression designed to detect sentence endings considering punctuation and abbreviations, storing it in `self.sentence_end_regex`. It returns `None`.

#### `split_into_sentences(self, text: str) -> List[str]`
**How it changes the document:** It breaks down a block of text into distinct, individual sentences.
**What information it provides for the RAG pipeline:** Provides the atomic text units (sentences) necessary for performing pairwise semantic boundary detection.
**Line-by-line explanation:** Lines 50-51 check if the stripped input `text` is empty, returning an empty list if it is. Line 53 splits the `text` using a regular expression that identifies whitespace preceded by sentence-ending punctuation (.!?). Line 54 returns a list comprising the stripped versions of the resulting sentence strings, filtering out any empty strings along the way.

#### `detect_boundaries(self, sentences: List[str]) -> List[Dict[str, Any]]`
**How it changes the document:** It analyzes the sentences to mark semantic shifts, effectively annotating where topics change without altering the text.
**What information it provides for the RAG pipeline:** Provides boundary markers and distance metrics indicating where the text should be split into coherent semantic chunks.
**Line-by-line explanation:** Lines 58-59 check if the `sentences` list has fewer than 2 elements, returning a default list indicating no boundaries if true. Line 61 obtains vector embeddings for all `sentences` using the provided `vector_engine`. Lines 63-68 iterate over consecutive sentence pairs, calculating their cosine similarity, deriving the distance (1 - similarity), storing it in the `distances` list, and maintaining a running mean. Line 70 calculates the standard deviation of all computed distances. Line 71 determines a `dynamic_threshold` by adding the mean distance to the product of the `threshold_factor` and the standard deviation. Lines 73-89 iterate through the `sentences`, comparing each sentence's distance to the next against the `dynamic_threshold` to set the `is_boundary` flag, and construct a list of result dictionaries containing the index, sentence string, distance, and boundary status, which is then returned.

#### `cluster_sentences(self, sentences: List[str], boundary_analysis: List[Dict[str, Any]]) -> List[str]`
**How it changes the document:** It groups individual sentences into larger, semantically coherent text chunks based on the provided boundaries.
**What information it provides for the RAG pipeline:** Provides the pipeline with fully assembled, contiguous text chunks ready for final embedding and retrieval.
**Line-by-line explanation:** Lines 92-93 initialize an empty list for `chunks` and a buffer list for `current_chunk_sentences`. Lines 95-99 iterate through the `sentences`, appending each to the current buffer; if the corresponding `boundary_analysis` indicates a boundary, the sentences in the buffer are joined with spaces, appended to the `chunks` list, and the buffer is cleared. Lines 101-102 check if any sentences are left in the buffer after the loop finishes, joining and appending them as a final chunk. Line 104 returns the fully assembled list of string `chunks`.

### Class: `SlidingSemanticChunker`

#### `__init__(self, vector_engine: SemanticEngine, window_size: int = 3, threshold_factor: float = 0.8)`
**How it changes the document:** It does not change the document.
**What information it provides for the RAG pipeline:** Initializes the sliding window chunker with a vector engine, a specified window size for context, and a threshold factor.
**Line-by-line explanation:** Lines 108-110 assign the provided `vector_engine`, `window_size`, and `threshold_factor` arguments to their respective instance variables on `self`. It returns `None`.

#### `split_into_sentences(self, text: str) -> List[str]`
**How it changes the document:** It breaks down a large text block into a list of individual sentences.
**What information it provides for the RAG pipeline:** Provides the fundamental sentence units required to build multi-sentence sliding windows for context-aware boundary detection.
**Line-by-line explanation:** Lines 113-114 verify if the stripped `text` is empty, returning an empty list if it is. Line 116 splits the text string using a regex that targets whitespace following punctuation marks (.!?). Line 117 returns a list of these stripped sentence strings, making sure to exclude any empty strings.

#### `compute_window_bounds(self, sentences: List[str]) -> List[Dict[str, Any]]`
**How it changes the document:** It analyzes the document context over sliding windows to mark robust semantic shifts without altering the underlying text.
**What information it provides for the RAG pipeline:** Provides highly accurate semantic boundary markers by analyzing broader contexts rather than just pairwise adjacent sentences.
**Line-by-line explanation:** Lines 120-122 check if there are enough `sentences` to form both a full left and right window; if not, it returns a list indicating no boundaries exist. Lines 124-125 initialize empty lists for left and right window strings. Line 127 identifies the `valid_split_indices` where a full sliding window can exist. Lines 129-134 loop over these valid indices, slicing out the left and right window sentence groups, joining them into single strings, and appending them to their respective lists. Lines 136-137 generate embeddings for all left and right window strings using the `vector_engine`. Lines 139-142 iterate through the valid indices again, calculating the cosine similarity between the left and right window embeddings and converting this to a distance metric. Lines 144-146 calculate the mean and standard deviation of these window distances to establish a `dynamic_threshold`. Lines 148-152 create a `boundary_map` dictionary initialized to `False` for all sentences, and set to `True` for valid indices where the window distance exceeds the `dynamic_threshold`. Lines 154-168 assemble an `analysis` list mapping each sentence to its distance (or 0.0 if not a valid split point) and its boolean boundary status. Line 170 returns this complete `analysis` list.

#### `generate_chunks(self, sentences: List[str], analysis: List[Dict[str, Any]]) -> List[str]`
**How it changes the document:** It restructures the document by grouping sequences of sentences into defined semantic chunks based on window boundaries.
**What information it provides for the RAG pipeline:** Provides the final, high-quality semantically cohesive text chunks that improve vector retrieval accuracy.
**Line-by-line explanation:** Lines 173-174 initialize an empty `chunks` list and a `buffer` list for the current chunk. Lines 176-180 iterate over the `sentences`, adding each to the `buffer`; if the `analysis` for that sentence index shows a boundary, the `buffer` is joined into a single string, appended to the `chunks` list, and the `buffer` is reset. Lines 182-183 handle any remaining sentences in the `buffer` after the loop by joining and appending them as a final chunk. Line 185 returns the finalized list of `chunks`.

---

## File: `markdown_parser.py`

### Class: `NodeType`
*(This class contains only string constants representing ROOT, HEADING, PARAGRAPH, TABLE, LIST_ITEM, and LIST.)*

### Class: `DocNode`

#### `__init__(self, node_type: NodeType, text: str, level: int = 0, metadata: Optional[Dict[str, Any]] = None)`
**How it changes the document:** It encapsulates raw text into a structured node format, forming the basis of a document tree.
**What information it provides for the RAG pipeline:** Provides the foundational building block for constructing a hierarchical representation of the document.
**Line-by-line explanation:** Line 20 generates a unique 8-character string from a UUID and assigns it to `self.node_id`. Lines 21-23 assign the provided `node_type`, `text`, and `level` arguments to `self`. Line 24 assigns the `metadata` dictionary to `self.metadata`, or an empty dictionary if `metadata` is `None`. Lines 26-27 initialize `self.parent` to `None` and `self.children` to an empty list. It returns a new `DocNode` instance.

#### `add_child(self, child_node: 'DocNode') -> None`
**How it changes the document:** It modifies the structural representation of the document by establishing a parent-child relationship between two nodes.
**What information it provides for the RAG pipeline:** Enables the construction of the hierarchical document tree, which is necessary to extract contextual heading paths later.
**Line-by-line explanation:** Line 30 assigns the `parent` attribute of the incoming `child_node` to reference `self`. Line 31 appends the `child_node` to the `self.children` list of the current node. It returns `None`.

#### `get_contextual_path(self) -> List[str]`
**How it changes the document:** It does not change the document.
**What information it provides for the RAG pipeline:** Provides an ordered sequence of heading text strings that denote the node's location within the document's structure, offering critical context for the chunks.
**Line-by-line explanation:** Line 34 initializes an empty `path` list. Line 35 initializes a `curr` variable pointing to `self` to start the traversal. Lines 36-39 execute a while loop that walks up the tree using the `parent` pointer until it encounters the `ROOT` node; if the current node is of type `HEADING`, its `text` is inserted at the beginning (index 0) of the `path` list. Line 40 returns the constructed `path` list of strings.

#### `__repr__(self) -> str`
**How it changes the document:** It does not change the document.
**What information it provides for the RAG pipeline:** Provides a human-readable string representation of the node for debugging and logging purposes.
**Line-by-line explanation:** Line 42 formats a string containing the node's ID, type, level, and the count of its children, and returns this formatted string.

### Class: `markdownParser`

#### `__init__(self)`
**How it changes the document:** It does not change the document.
**What information it provides for the RAG pipeline:** Initializes the parser with the regular expressions needed to interpret and structure raw markdown text.
**Line-by-line explanation:** Lines 46-48 compile regular expressions for identifying markdown headings, list items, and table rows, and store them in `self.heading_regex`, `self.list_regex`, and `self.table_row_regex` respectively. It returns `None`.

#### `_flush_table(self, parent: DocNode, buffer: List[str], start_line: int) -> None`
**How it changes the document:** It groups accumulated sequential table rows into a single, unified table node within the document tree.
**What information it provides for the RAG pipeline:** Provides intact table chunks to the pipeline, preserving tabular row and column structures without semantically splitting them.
**Line-by-line explanation:** Line 56 joins the strings inside the `buffer` list using newlines to create a unified `table_text` string. Lines 57-62 construct a new `DocNode` with type `TABLE`, the combined text, the `parent`'s level, and a metadata dictionary containing the starting line number and total row count. Line 63 invokes `parent.add_child` to attach this new table node to the document tree. Line 64 clears the `buffer` to ready it for any future tables. It returns `None`.

#### `_flush_paragraph(self, parent: DocNode, buffer: List[str], start_line: int) -> None`
**How it changes the document:** It groups lines of a paragraph together.
**What information it provides for the RAG pipeline:** Accumulates normal text paragraphs as a single node in the document tree.
**Line-by-line explanation:** Joins buffer lines using a space to construct continuous prose text, instantiates a `NodeType.PARAGRAPH` node with starting line metadata, appends it as a child to `parent`, and clears the buffer.

#### `_flush_list(self, parent: DocNode, buffer: List[str], start_line: int) -> None`
**How it changes the document:** It groups consecutive list item lines into a single LIST node.
**What information it provides for the RAG pipeline:** Prevents list item fracturing, keeping lists intact for coherent semantic retrieval.
**Line-by-line explanation:** Joins accumulated list lines using a newline (preserving indentations and bullet formats), instantiates a `NodeType.LIST` node, appends it to `parent`, and clears the buffer.

#### `parse(self, text: str) -> DocNode`
**How it changes the document:** It transforms flat, unstructured markdown text into a robust hierarchical node tree representation.
**What information it provides for the RAG pipeline:** Provides the complete structural skeleton of the entire document, setting the stage for context-aware chunking.
**Line-by-line explanation:** Instantiates a `ROOT` `DocNode` to serve as the tree's base. It iterates over each line, buffering tables, paragraphs, and lists. Blank lines, new headings, or other constructs trigger the flushing of open buffers (via `_flush_table`, `_flush_paragraph`, or `_flush_list`). Heading matches traverse up the lineage to establish hierarchy based on hash levels. List items are grouped consecutively into the list buffer. The method returns the root node representing the structured document tree.

---

## File: `hierarchical_chunker.py`

### Class: `RenderedChunk`

#### `__init__(self, chunk_id: str, parent_node_id: str, text: str, metadata: Dict[str, Any])`
**How it changes the document:** It does not alter the original document, but represents its final chunked form.
**What information it provides for the RAG pipeline:** Provides the pipeline with a fully processed, context-enriched data chunk that is ready to be embedded and stored in a vector database.
**Line-by-line explanation:** Lines 9-12 take the provided `chunk_id`, `parent_node_id`, the context-enriched `text`, and the `metadata` dictionary and assign them to instance variables of the same names on `self`. It returns `None`.

#### `__repr__(self) -> str`
**How it changes the document:** It does not change the document.
**What information it provides for the RAG pipeline:** Provides a compact string representation of the rendered chunk for debugging.
**Line-by-line explanation:** Line 15 formats and returns a string displaying the chunk's ID, its parent node ID, and the integer length of its text content.

### Class: `HierarchicalSemanticEngine`

#### `__init__(self, vector_engine: SemanticEngine, window_size: int = 3, threshold_factor: float = 0.8)`
**How it changes the document:** It does not change the document.
**What information it provides for the RAG pipeline:** Initializes the primary orchestration engine that ties document structure and semantic boundaries together.
**Line-by-line explanation:** Line 20 creates an instance of `markdownParser` and stores it in `self.structure_parser`. Lines 21-24 instantiate a `SlidingSemanticChunker` with the provided `vector_engine`, `window_size`, and `threshold_factor`, and assign it to `self.boundary_detector`. It returns `None`.

#### `process_document(self, raw_markdown_text: str, source_name: str = "unknown") -> List[RenderedChunk]`
**How it changes the document:** It takes raw markdown text, builds a structure tree, decomposes it into chunks, and enriches them with hierarchical paths.
**What information it provides for the RAG pipeline:** Provides the complete, end-to-end execution resulting in a final list of `RenderedChunk` objects that have both semantic cohesion and structural context.
**Line-by-line explanation:** Line 28 invokes the `structure_parser.parse` method on the `raw_markdown_text` to generate a root `DocNode`. Line 30 sets up an empty list called `final_chunks` to collect results. Line 31 calls the internal `_decompose_node` recursive method, passing it the root node, the `final_chunks` accumulator, and the `source_name`. Line 33 returns the populated `final_chunks` list containing all rendered chunks.

#### `_decompose_node(self, node: DocNode, chunk_accumulator: List[RenderedChunk], source_name: str) -> None`
**How it changes the document:** It dissects a document node into semantic chunks and injects its structural heading path directly into the chunk's text representation.
**What information it provides for the RAG pipeline:** Provides the core processing logic that ensures every text chunk sent to the vector database retains self-explanatory structural context.
**Line-by-line explanation:** Traverses the AST recursively. PARAGRAPH nodes are split using `SlidingSemanticChunker` sentence clustering and token limits. LIST nodes are treated as cohesive semantic blocks, processed as a unit, and safely split only if exceeding max token limits. TABLE nodes are converted into structured block text. All chunks are enriched with contextual breadcrumbs (e.g., `Context: H1 > H2`) prepended to the text, and appended to the final chunk accumulator.
