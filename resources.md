### Semantic Search

Semantic search seeks to improve search accuracy by understanding the content of the search query. In contrast to traditional search engines which only find documents based on lexical matches, semantic search can also find synonyms.

1. Why semantic search in LLaMa Index? 
   - LLM cannot support large tokens, so we need to use semantic search to find similar documents or paragraphs. This strategy can help us reduce the number of tokens to a reasonable size.
   - Because it is a very powerful tool to find similar documents. It is based on the semantic similarity of the documents, not on the lexical similarity. This means that it can find documents that are not lexically similar but are semantically similar. For example, if you search for "car", it will find documents that contain "automobile" or "vehicle". This is very useful for finding documents that are not lexically similar but are semantically similar.

2. Why use HuggingFaceEmbeddings?
The handy model overview table in [the documentation](https://www.sbert.net/docs/pretrained_models.html) indicates that the `sentence-transformers/all-mpnet-base-v2` checkpoint has the best performance for semantic search, so we’ll use that for our application.

3. What's the reranking and why we need it?

More details on:
- https://www.sbert.net/examples/applications/semantic-search/README.html
- https://rom1504.medium.com/semantic-search-with-embeddings-index-anything-8fb18556443c
- https://vectara.com/what-is-reranking-and-why-does-it-matter/

### Performance and Benchmarking

### Why define vector index and keyword table index over the same publications?

https://gpt-index.readthedocs.io/en/latest/examples/query_engine/CustomRetrievers.html#define-vector-index-and-keyword-table-index-over-same-data

### How to improve the performance of the semantic search?

