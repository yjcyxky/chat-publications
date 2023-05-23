import os
from typing import Optional, List, Sequence, Set, Any

from qdrant_client.models import Distance, VectorParams

from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.data_structs import Node, NodeWithScore
from llama_index.data_structs.data_structs import KeywordTable

from llama_index.indices.keyword_table.base import BaseGPTKeywordTableIndex
from llama_index.indices.keyword_table.utils import simple_extract_keywords
from llama_index.prompts.prompts import KeywordExtractPrompt
from llama_index.readers.file.tabular_parser import PandasCSVParser
from llama_index.readers.file.base import DEFAULT_FILE_EXTRACTOR
from llama_index.indices.postprocessor.cohere_rerank import CohereRerank
from llama_index.response.pprint_utils import pprint_response
from llama_index.storage.docstore.mongo_docstore import MongoDocumentStore
from llama_index.storage.index_store import MongoIndexStore

from llama_index.retrievers import BaseRetriever, VectorIndexRetriever, KeywordTableSimpleRetriever
from llama_index import (QueryBundle, ServiceContext, StorageContext,
                         QuestionAnswerPrompt, load_indices_from_storage)

class FilterNodes(BaseNodePostprocessor):
    def __init__(
        self,
        similarity: int = 0.8,
    ):
        self.similarity = similarity

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        new_nodes = []
        for node in nodes:
            print(f"\n\n-----------\nFiltering nodes with similarity:\n[Content]: \n{node.source_text} \n \n[Score]: {node.score}\n-----------")
            if node.score is not None and node.score >= self.similarity:
                new_nodes.append(node)

            # Maybe the score is None when KeywordTableIndex is used
            if node.score is None:
                new_nodes.append(node)
                
        return new_nodes


class CustomKeywordTableIndex(BaseGPTKeywordTableIndex):
    """Custom Keyword Table Index.

    This index uses manually defined keywords to index into the keyword table.

    keywords: List[List[str]], a list of keywords for each text chunk

    """

    def __init__(self, nodes: Sequence[Node] | None = None, index_struct: KeywordTable | None = None, service_context: ServiceContext | None = None, keyword_extract_template: KeywordExtractPrompt | None = None, max_keywords_per_chunk: int = 10, use_async: bool = False, keywords: Optional[List[List[str]]] = [], **kwargs: Any) -> None:
        self.keywords = keywords
        super().__init__(nodes, index_struct, service_context,
                         keyword_extract_template, max_keywords_per_chunk, use_async, **kwargs)

    def _add_nodes_to_index(
        self, index_struct: KeywordTable, nodes: Sequence[Node]
    ) -> None:
        """Add document to index."""
        for (idx, n) in enumerate(nodes):
            if self.keywords:
                keywords = set(self.keywords[idx])
            else:
                keywords = self._extract_keywords(n.get_text())
            index_struct.add_node(list(keywords), n)

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        return simple_extract_keywords(text, self.max_keywords_per_chunk)
    

class CustomPandasCSVParser(PandasCSVParser):
    def __init__(self, *args: Any, concat_rows: bool = True, col_joiner: str = ", ", row_joiner: str = "\n", pandas_config: dict = ..., **kwargs: Any) -> None:
        super().__init__(*args, concat_rows=concat_rows, col_joiner=col_joiner,
                         row_joiner=row_joiner, pandas_config=pandas_config, **kwargs)

        self._pandas_config = self._pandas_config.update({"delimiter": "\t"})


CUSTOM_FILE_READER_CLS = {
    **DEFAULT_FILE_EXTRACTOR,
    "tsv": CustomPandasCSVParser
}


def get_qdrant_store(persist_dir=None, collection_name="pubmed", create_collection=False):
    import qdrant_client
    from llama_index.vector_stores import QdrantVectorStore
    # Creating a Qdrant vector store
    if persist_dir:
        client = qdrant_client.QdrantClient(path=persist_dir)
    else:
        client = qdrant_client.QdrantClient(host="localhost", port=6333)

        if create_collection:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

    # construct vector store
    store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
    )
    return store


def get_custom_qa_prompt():
    QA_PROMPT_TMPL = (
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please answer the question: {query_str}\n"
    )
    QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

    return QA_PROMPT


class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "AND"
    ) -> None:
        """Init params."""

        print("Initializing custom retriever.")
        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.get_doc_id() for n in vector_nodes}
        keyword_ids = {n.node.get_doc_id() for n in keyword_nodes}

        combined_dict = {n.node.get_doc_id(): n for n in vector_nodes}
        combined_dict.update({n.node.get_doc_id(): n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        print(f"Retrieved {len(retrieve_nodes)} nodes.")
        return retrieve_nodes


def get_comp_query_engine(vector_index=None, keyword_index=None, service_context=None, 
                          mode="OR", similarity_top_k=10, node_postprocessors=None,
                          text_qa_template=None, **kwargs):
    from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine

    vector_retriever = None
    keyword_retriever = None
    # define custom retriever
    if vector_index:
        print("Initializing custom retriever (vector).")
        vector_retriever = VectorIndexRetriever(
            index=vector_index, similarity_top_k=similarity_top_k
        )
    
    if keyword_index:
        print("Initializing custom retriever (keyword).")
        keyword_retriever = KeywordTableSimpleRetriever(
            index=keyword_index, num_chunks_per_query=similarity_top_k
        )

    if vector_retriever and keyword_retriever:
        print("Initializing custom retriever (vector + keyword).")
        custom_retriever = CustomRetriever(vector_retriever, keyword_retriever, mode=mode)
    elif vector_retriever:
        custom_retriever = vector_retriever
    elif keyword_retriever:
        custom_retriever = keyword_retriever
    else:
        raise ValueError("Must provide at least one index.")

    # assemble query engine
    print("Assembling query engine.")
    custom_query_engine = RetrieverQueryEngine.from_args(
        service_context=service_context,
        retriever=custom_retriever,
        node_postprocessors=node_postprocessors,
        text_qa_template=text_qa_template,
        **kwargs,
    )

    return custom_query_engine


def get_storage_context(persist_dir, index_type, collection_name="pubmed", create_collection=False):
    # rebuild storage context
    if index_type == "default":
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    elif index_type == "qdrant":
        store = get_qdrant_store(
            persist_dir, collection_name=collection_name, 
            create_collection=create_collection
        )
        storage_context = StorageContext.from_defaults(
            vector_store=store,
            persist_dir=persist_dir
        )
    elif index_type == "qdrant-prod":
        mongodb_docstore = MongoDocumentStore.from_host_and_port(host="localhost", port=27017)
        mongodb_index_store = MongoIndexStore.from_host_and_port(host="localhost", port=27017)
        store = get_qdrant_store(collection_name=collection_name, create_collection=create_collection)
        storage_context = StorageContext.from_defaults(
            docstore=mongodb_docstore,
            index_store=mongodb_index_store,
            vector_store=store,
            persist_dir=persist_dir
        )
    else:
        raise ValueError(f"Invalid index_type: {index_type}")

    return storage_context


def get_query_engine(persist_dir, index_type, service_context, index_id=None, 
                     collection_name="pubmed", **kwargs):
    storage_context = get_storage_context(persist_dir, index_type, collection_name=collection_name)

    indices = load_indices_from_storage(
        storage_context,
        service_context=service_context
    )

    if len(indices) == 1:
        print("Using vector index only")
        doc_vector_index = indices[0]
        query_engine = doc_vector_index.as_query_engine(**kwargs)
    else:
        doc_vector_index, keyword_table_index = indices
        if index_id == 'doc_vector_index':
            print("Using vector index only")
            doc_vector_index = indices[0]
            query_engine = get_comp_query_engine(
                vector_index=doc_vector_index, keyword_index=None,
                service_context=service_context,
                **kwargs
            )
        elif index_id == 'keyword_table_index':
            print("Using keyword table index only")
            query_engine = get_comp_query_engine(
                vector_index=None, keyword_index=keyword_table_index,
                service_context=service_context,
                **kwargs
            )
        else:
            print("Using hybrid index")
            query_engine = get_comp_query_engine(
                vector_index=doc_vector_index, keyword_index=keyword_table_index, 
                service_context=service_context,
                **kwargs
            )

    return query_engine

def get_chatbot(similarity=0.8, similarity_top_k=10, persist_dir=os.getcwd(), index_type="default", 
                service_context=None, index_id=None, collection_name="pubmed"):
    print("Loading indexes...")
    qa_prompt = get_custom_qa_prompt()
    # add postprocessor
    # Remove nodes with similarity < 0.9
    filter_nodes_with_similarity = FilterNodes(similarity=similarity)

    api_key = os.environ.get("COHERE_API_KEY")
    if api_key:
        print("Using CohereRerank...")
        cohere_rerank = CohereRerank(api_key=api_key, top_n=similarity_top_k)
        query_engine = get_query_engine(persist_dir=persist_dir, index_type=index_type,
                                        service_context=service_context,
                                        similarity_top_k=similarity_top_k * 2,
                                        node_postprocessors=[
                                            cohere_rerank, filter_nodes_with_similarity
                                        ],
                                        text_qa_template=qa_prompt,
                                        index_id=index_id,
                                        collection_name=collection_name)
    else:
        print("Using default results...")
        query_engine = get_query_engine(persist_dir=persist_dir, index_type=index_type,
                                        service_context=service_context,
                                        similarity_top_k=similarity_top_k,
                                        node_postprocessors=[
                                            filter_nodes_with_similarity
                                        ],
                                        text_qa_template=qa_prompt,
                                        index_id=index_id,
                                        collection_name="pubmed")
        
    def remove_redundant_blank(text):
        return " ".join(text.split())

    def chatbot(input_text):
        print("Input: %s" % input_text)
        response = query_engine.query(input_text)
        pprint_response(response)
        nodes_extra_info = [node.extra_info for node in response.source_nodes]
        extra_info = [remove_redundant_blank(f'{index + 1}. {node.get("authors")} {node.get("title")}. {node.get("journal")}, \
                        {node.get("country")} {node.get("pubdate")}. \
                        [DOI: {node.get("doi")}] [PMID: {node.get("pmid")}]')
                      for (index, node) in enumerate(nodes_extra_info)]
        references = "\n".join(extra_info)
        if response.response is None:
            return "Don't know the answer (cannot find the related context information from the knowledge base.)"
        return f"""{response.response.strip()}\n\nReferences: \n{references}""".strip()

    return chatbot