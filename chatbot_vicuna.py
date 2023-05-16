import openai
import os
import re
import click
import gradio as gr

from typing import Callable, Dict, Optional, List, Mapping, Any

from langchain.llms.base import LLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext, LangchainEmbedding, PromptHelper
from llama_index.readers.file.tabular_parser import PandasCSVParser
from llama_index.readers.file.base import DEFAULT_FILE_EXTRACTOR
from llama_index import StorageContext, load_index_from_storage
from llama_index.response.pprint_utils import pprint_response
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.indices.postprocessor.cohere_rerank import CohereRerank

OPENAI_API_KEY = "EMPTY"  # Not support yet
OPENAI_API_BASE = "http://localhost:8000/v1"

os.environ['HF_HOME'] = str(os.getcwd()) + '/huggingface'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:512"

# define prompt helper
# set maximum input size
max_input_size = 2048
# set number of output tokens
num_output = 512
# set maximum chunk overlap
max_chunk_overlap = 0


class CustomPandasCSVParser(PandasCSVParser):
    def __init__(self, *args: Any, concat_rows: bool = True, col_joiner: str = ", ", row_joiner: str = "\n", pandas_config: dict = ..., **kwargs: Any) -> None:
        super().__init__(*args, concat_rows=concat_rows, col_joiner=col_joiner,
                         row_joiner=row_joiner, pandas_config=pandas_config, **kwargs)

        self._pandas_config = self._pandas_config.update({"delimiter": "\t"})


CUSTOM_FILE_READER_CLS = {
    **DEFAULT_FILE_EXTRACTOR,
    "tsv": CustomPandasCSVParser
}


class CustomHttpLLM(LLM):
    model_name = "vicuna-13b"

    def model_pipeline(self, prompt: str) -> str:
        completion = openai.ChatCompletion.create(
            api_key=OPENAI_API_KEY,
            api_base=OPENAI_API_BASE,
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

    def __init__(self):
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
                                       max_length=max_input_size)

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


def launch_chatbot(persist_dir, llm_type="custom"):
    service_context = get_service_context(llm_type)
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    # load index
    index = load_index_from_storage(storage_context, service_context=service_context)
    api_key = os.environ["COHERE_API_KEY"]
    if api_key:
        print("Using CohereRerank...")
        cohere_rerank = CohereRerank(api_key=api_key, top_n=2)
        index = index.as_query_engine(similarity_top_k=10, node_postprocessors=[cohere_rerank])
    else:
        print("Using default results...")
        index = index.as_query_engine(similarity_top_k=2)

    def chatbot(input_text):
        print("Input: %s" % input_text)
        response = index.query(input_text)
        pprint_response(response)
        return response.response.strip()

    return chatbot


@click.group()
def chatbot():
    pass


def get_service_context(llm_type="custom"):
    if llm_type == "custom":
        llm_predictor = LLMPredictor(llm=CustomLLM())
    elif llm_type == "custom-http":
        llm_predictor = LLMPredictor(llm=CustomHttpLLM())
    else:
        raise ValueError(f"Invalid llm_type: {llm_type}")

    embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

    node_parser = SimpleNodeParser(text_splitter=TokenTextSplitter(
        chunk_size=512, chunk_overlap=max_chunk_overlap))
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, embed_model=embed_model,
        prompt_helper=prompt_helper, node_parser=node_parser, chunk_size_limit=512
    )
    return service_context


@chatbot.command(help="Build index from directory of documents.")
@click.option('--directory-path', '-d', required=True, help="The directory which saved the documents.")
@click.option('--llm-type', '-l', default="custom", help="The type of language model.", type=click.Choice(["custom", "custom-http"]))
@click.option('--mode', '-m', default="node", help="The mode of indexing.", type=click.Choice(["node", "default"]))
def index(directory_path, llm_type, mode):
    service_context = get_service_context(llm_type)
    if mode == "node":
        import uuid
        from llama_index.data_structs.node import Node
        nodes = []
        for file in os.listdir(directory_path):
            # Treat each .txt file as a node, and the content of the file as the text of the node.
            # So we can load the whole file as the context of query. It maybe a good idea when you want to
            # search the answer from related single publicaion.
            if file.endswith(".txt"):
                uuid_str = str(uuid.uuid4())
                with open(os.path.join(directory_path, file), "r") as f:
                    text = f.read()
                    node = Node(text=text, doc_id=uuid_str)
                    nodes.append(node)
        doc_index = GPTVectorStoreIndex(nodes, service_context=service_context)
    else:
        documents = SimpleDirectoryReader(
            directory_path, file_extractor=CUSTOM_FILE_READER_CLS
        ).load_data()

        doc_index = GPTVectorStoreIndex.from_documents(
            documents, service_context=service_context
        )

    dirname = os.path.dirname(directory_path)
    doc_index.storage_context.persist(persist_dir=dirname)


@chatbot.command(help="Query index.")
@click.option('--directory-path', '-d', required=True, help="The directory which saved the documents.")
@click.option('--llm-type', '-l', default="custom", help="The type of language model.", type=click.Choice(["custom", "custom-http"]))
def query(directory_path, llm_type):
    if os.path.exists(directory_path):
        iface = gr.Interface(fn=launch_chatbot(directory_path, llm_type=llm_type),
                             inputs=gr.inputs.Textbox(lines=7,
                                                      label="Enter your text"),
                             outputs="text",
                             title="Custom-trained AI Chatbot")

        iface.queue().launch(debug=True, share=True, inline=False)
    else:
        print("Index file not found.")
        return


if __name__ == "__main__":
    chatbot()
