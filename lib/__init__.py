import os
import re
import openai
from typing import Callable, Dict, Optional, List, Mapping, Any, Sequence, Set
from llama_index.data_structs import Node, NodeWithScore
from llama_index.data_structs.data_structs import KeywordTable

from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.indices.keyword_table.base import BaseGPTKeywordTableIndex
from llama_index.indices.keyword_table.utils import simple_extract_keywords

from llama_index import (QueryBundle, LLMPredictor, ServiceContext,
                         LangchainEmbedding, PromptHelper, StorageContext,
                         QuestionAnswerPrompt, load_indices_from_storage)
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.prompts.prompts import KeywordExtractPrompt
from langchain.llms.base import LLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.readers.file.tabular_parser import PandasCSVParser
from llama_index.readers.file.base import DEFAULT_FILE_EXTRACTOR

from llama_index.retrievers import BaseRetriever, VectorIndexRetriever, KeywordTableSimpleRetriever


def get_service_context(llm_type="custom", max_chunk_overlap=0.5,
                        max_input_size=1500, num_output=512,
                        chunk_size_limit=512, openai_api_key=None,
                        openai_api_base=None) -> ServiceContext:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["OPENAI_API_BASE"] = openai_api_base
    os.environ["MAX_INPUT_SIZE"] = str(max_input_size)
    print(f"Setting up service context with llm_type: {llm_type}")
    if llm_type == "custom":
        llm_predictor = LLMPredictor(
            llm=CustomLLM()
        )
    elif llm_type == "custom-http":
        llm_predictor = LLMPredictor(llm=CustomHttpLLM())
    else:
        raise ValueError(f"Invalid llm_type: {llm_type}")

    embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

    node_parser = SimpleNodeParser(text_splitter=TokenTextSplitter(
        chunk_size=chunk_size_limit, chunk_overlap=max_chunk_overlap)
    )
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, embed_model=embed_model,
        prompt_helper=prompt_helper, node_parser=node_parser,
        chunk_size_limit=chunk_size_limit
    )
    return service_context


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


class FilterNodes(BaseNodePostprocessor):
    def __init__(
        self,
        similarity: int = 0.8
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


class CustomHttpLLM(LLM):
    model_name = "vicuna-13b"

    def model_pipeline(self, prompt: str) -> str:
        completion = openai.ChatCompletion.create(
            api_key=os.environ["OPENAI_API_KEY"],
            api_base=os.environ["OPENAI_API_BASE"],
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content

    def remove_html_tags(self, text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print(f"{prompt}, {type(prompt)}")
        res = self.model_pipeline(str(prompt))
        try:
            return res
        except Exception as e:
            print(e)
            return "Don't know the answer"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"


class CustomLLM(LLM):
    # model_name = "eachadea/vicuna-13b-1.1"
    model_name = "lmsys/vicuna-7b-delta-v1.1"

    def __init__(self, *args, **kwargs):
        import torch
        from transformers import pipeline
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = 'cpu'

        device = torch.device(device)
        print(f"Using device: {device}")

        self.model_pipeline = pipeline("text-generation", model=self.model_name, device_map='auto',
                                       trust_remote_code=True, model_kwargs={"torch_dtype": torch.bfloat16, "load_in_8bit": True},
                                       max_length=MAX_INPUT_SIZE)

    def remove_html_tags(self, text):
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print(prompt, type(prompt))
        res = self.model_pipeline(str(prompt))
        print(res, type(res))
        if len(res) >= 1:
            generated_text = res[0].get("generated_text")[len(prompt):]
            return generated_text
        else:
            return "Don't know the answer"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"


def get_qdrant_store(persist_dir):
    import qdrant_client
    from llama_index.vector_stores import QdrantVectorStore
    # Creating a Qdrant vector store
    client = qdrant_client.QdrantClient(path=persist_dir)
    collection_name = "vicuna"

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
        "Given this information, please answer the question and list all related publications as references: {query_str}\n"
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
        return retrieve_nodes


def get_comp_query_engine(vector_index, keyword_index, service_context=None, 
                          mode="OR", similarity_top_k=10, node_postprocessors=None,
                          text_qa_template=None, **kwargs):
    from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine

    # define custom retriever
    vector_retriever = VectorIndexRetriever(
        index=vector_index, similarity_top_k=similarity_top_k
    )
    keyword_retriever = KeywordTableSimpleRetriever(index=keyword_index)
    custom_retriever = CustomRetriever(vector_retriever, keyword_retriever, mode=mode)

    # assemble query engine
    custom_query_engine = RetrieverQueryEngine.from_args(
        service_context=service_context,
        retriever=custom_retriever,
        node_postprocessors=node_postprocessors,
        text_qa_template=text_qa_template,
        **kwargs,
    )

    return custom_query_engine


def get_query_engine(persist_dir, index_type, service_context, **kwargs):
    # rebuild storage context
    if index_type == "default":
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    elif index_type == "qdrant":
        store = get_qdrant_store(persist_dir)
        storage_context = StorageContext.from_defaults(
            vector_store=store,
            persist_dir=persist_dir
        )

    indices = load_indices_from_storage(
        storage_context,
        service_context=service_context
    )

    if len(indices) == 1:
        print("Using vector index only")
        doc_vector_index = indices[0]
        query_engine = doc_vector_index.as_query_engine(**kwargs)
    else:
        print("Using hybrid index")
        doc_vector_index, keyword_table_index = indices
        query_engine = get_comp_query_engine(
            doc_vector_index, keyword_table_index, 
            service_context=service_context,
            **kwargs
        )

    return query_engine
