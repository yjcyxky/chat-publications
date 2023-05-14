import os
import re
import torch
import click
import gradio as gr

from transformers import pipeline
from langchain.llms.base import LLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, GPTListIndex, GPTKeywordTableIndex, Document, MockEmbedding, LLMPredictor, ServiceContext, LangchainEmbedding, PromptHelper
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser.simple import SimpleNodeParser
from typing import Optional, List, Mapping, Any
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

os.environ['HF_HOME'] = str(os.getcwd()) + '/huggingface'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:512"
# define prompt helper
# set maximum input size
max_input_size = 2048
# set number of output tokens
num_output = 512
# set maximum chunk overlap
max_chunk_overlap = 20

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = 'cpu'

device = torch.device(device)
print(f"Using device: {device}")


class CustomLLM(LLM):
    # model_name = "eachadea/vicuna-13b-1.1"
    model_name = "lmsys/vicuna-7b-delta-v1.1"
    model_pipeline = pipeline("text-generation", model=model_name, device_map = 'auto', 
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


def launch_chatbot(index_filepath="index.json", service_context=None):
    index = GPTSimpleVectorIndex.load_from_disk(
        index_filepath, service_context=service_context)

    def chatbot(input_text):
        response = index.query(input_text)
        return response.response.strip()

    return chatbot


@click.group()
def chatbot():
    pass


def get_service_context():
    llm_predictor = LLMPredictor(llm=CustomLLM())
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

    node_parser = SimpleNodeParser(text_splitter=TokenTextSplitter(
        chunk_size=512, chunk_overlap=max_chunk_overlap))
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, embed_model=embed_model, prompt_helper=prompt_helper, node_parser=node_parser, chunk_size_limit=512)
    return service_context


@chatbot.command(help="Build index from directory of documents.")
@click.option('--directory-path', '-d', required=True, help="The directory which saved the documents.")
def index(directory_path):
    service_context = get_service_context()
    documents = SimpleDirectoryReader(directory_path).load_data()

    doc_index = GPTSimpleVectorIndex.from_documents(
        documents, service_context=service_context)

    doc_index.save_to_disk(os.path.join(directory_path, 'index-vicuna.json'))

    return index


@chatbot.command(help="Query index.")
@click.option('--index-filepath', '-i', required=True, help="The index file path.")
def query(index_filepath):
    service_context = get_service_context()
    if os.path.exists(index_filepath):
        iface = gr.Interface(fn=launch_chatbot(index_filepath=index_filepath, service_context=service_context),
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
