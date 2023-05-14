import gradio as gr
import click
import json
import os
from llama_index import SimpleDirectoryReader, ServiceContext, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI

# os.environ["OPENAI_API_KEY"] = "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"
model_name = "text-davinci-003"
os.environ["OPENAI_API_KEY"] = "EMPTY" # Not support yet
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
# model_name = "vicuna-13b"


def check_variable():
    if os.environ.get("OPENAI_API_KEY") is None:
        print("Please set the OPENAI_API_KEY environment variable.")
        exit(1)


def launch_chatbot(index_filepath="index.json"):
    index = GPTSimpleVectorIndex.load_from_disk(index_filepath)

    def chatbot(input_text):
        response = index.query(input_text, response_mode="compact")
        return response.response.strip()

    return chatbot


@click.group()
def chatbot():
    pass


@chatbot.command(help="Build index from directory of documents.")
@click.option('--directory-path', '-d', required=True, help="The directory which saved the documents.")
def index(directory_path):
    check_variable()

    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(
        max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(
        temperature=0.7, model_name=model_name, max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    doc_index = GPTSimpleVectorIndex.from_documents(
        documents, service_context=service_context)
    doc_index = GPTSimpleVectorIndex.from_documents(documents)

    doc_index.save_to_disk(os.path.join(directory_path, 'index.json'))
    metadata = {
        "max_input_size": max_input_size,
        "num_outputs": num_outputs,
        "max_chunk_overlap": max_chunk_overlap,
        "chunk_size_limit": chunk_size_limit,
        "directory_path": directory_path,
        "index_type": "GPTSimpleVectorIndex",
        "model_name": "text-davinci-003",
        "temperature": 0.7,
        "max_tokens": num_outputs,
        "index_filepath": os.path.join(directory_path, 'index.json'),
        "num_documents": len(documents),
        "document_names": [os.path.basename(file) for file in os.listdir(directory_path)]
    }
    with open(os.path.join(directory_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)

    return index


@chatbot.command(help="Query index.")
@click.option('--index-filepath', '-i', required=True, help="The index file path.")
def query(index_filepath):
    check_variable()

    if os.path.exists(index_filepath):
        iface = gr.Interface(fn=launch_chatbot(index_filepath=index_filepath),
                             inputs=gr.inputs.Textbox(lines=7,
                                                      label="Enter your text"),
                             outputs="text",
                             title="Custom-trained AI Chatbot")

        iface.launch(share=False)
    else:
        print("Index file not found.")
        return


if __name__ == "__main__":
    chatbot()
