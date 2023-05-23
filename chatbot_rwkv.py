####################################################################################
import logging
from llama_index.indices.postprocessor.cohere_rerank import CohereRerank
from llama_index.response.pprint_utils import pprint_response
from llama_index import StorageContext
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
import gradio as gr
import click

import sys
import os
# Add lib to sys path
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
from lib import (get_qdrant_store, get_chatbot,
                 CustomKeywordTableIndex, CUSTOM_FILE_READER_CLS,)
from lib.rwkv import get_service_context
####################################################################################

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ['HF_HOME'] = str(os.getcwd()) + '/huggingface'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:512"

# define prompt helper
# set maximum input size
max_input_size = 5000
# set number of output tokens
num_output = 512
# set maximum chunk overlap
max_chunk_overlap = 0
# chunk size limit
chunk_size_limit = 512


def get_service_context_by_llm_type():
    service_context = get_service_context(
        max_input_size=max_input_size, num_output=num_output,
        max_chunk_overlap=max_chunk_overlap, chunk_size_limit=chunk_size_limit
    )

    return service_context


def launch_chatbot(persist_dir, index_type="default", similarity=0.9, 
                   index_id=None, similarity_top_k=5):
    service_context = get_service_context_by_llm_type()
    return get_chatbot(persist_dir=persist_dir, service_context=service_context, index_type=index_type,
                       similarity=similarity, index_id=index_id, similarity_top_k=similarity_top_k,
                       collection_name="pubmed")

@click.group()
def chatbot():
    pass


@chatbot.command(help="Build index from directory of documents.")
@click.option('--directory-path', '-d', required=True, help="The directory which saved the documents.")
@click.option('--mode', '-m', default="node", help="The mode of indexing.", type=click.Choice(["node", "default"]))
@click.option('--index-type', '-i', default="default", help="The type of index.", type=click.Choice(["default", "qdrant", "qdrant-prod"]))
@click.option('--persist-dir', '-p', default=os.getcwd(), help="The directory which saved the index.")
def index(directory_path, mode, index_type, persist_dir):
    service_context = get_service_context_by_llm_type()
    if index_type == "default":
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    elif index_type == "qdrant":
        store = get_qdrant_store(persist_dir)
        storage_context = StorageContext.from_defaults(
            vector_store=store
        )
    elif index_type == "qdrant-prod":
        store = get_qdrant_store()
        storage_context = StorageContext.from_defaults(
            vector_store=store
        )
    else:
        raise ValueError(f"Invalid index_type: {index_type}")

    def replace_all_chars(text, chars=['\n'], replacement=" "):
        for char in chars:
            text = text.replace(char, replacement)
        return text

    # We need to build three types of node:
    # - by title + abstract + pmid [Vector Store Index]
    # - by keywords + title + abstract + pmid [Keyword Table]
    # - by title + review full text [Tree Index]
    if mode == "node":
        import uuid
        import json
        from llama_index.data_structs.node import Node
        nodes = []
        keywords_lst = []
        # TODO: When the number of documents is large, we should use a more efficient way to load the data.
        for file in os.listdir(directory_path):
            # Treat each .txt file as a node, and the content of the file as the text of the node.
            # So we can load the whole file as the context of query. It maybe a good idea when you want to
            # search the answer from related single publicaion.
            if file.endswith(".json"):
                print(f"Loading {file}")
                with open(os.path.join(directory_path, file), "r") as f:
                    data = json.load(f)
                    for row in data:
                        if not row["title"] or not row["abstract"]:
                            continue

                        uuid_str = str(uuid.uuid4())
                        keys = list(row.keys())
                        keywords = row["keywords"].split(";")
                        keywords_lst.append(
                            [keyword.strip() for keyword in keywords]
                        )
                        content = [
                            f"{key}: {replace_all_chars(str(row[key]))}" for key in keys
                            if key in ["title", "abstract", "keywords"]]
                        extra_info = {
                            key: row[key]
                            for key in keys if key in ["pmid", "doi", "country", "journal", "pubdate"]
                        }

                        content = "\n".join(content)
                        node = Node(text=content, doc_id=uuid_str,
                                    extra_info=extra_info)
                        nodes.append(node)

        print("Building index...")
        doc_index = GPTVectorStoreIndex(
            nodes, service_context=service_context,
            storage_context=storage_context
        )
        doc_index.set_index_id("doc_vector_index")

        keyword_index = CustomKeywordTableIndex(
            nodes,
            keywords=keywords_lst,
            service_context=service_context,
            storage_context=storage_context
        )
        keyword_index.set_index_id("keyword_table_index")
    else:
        print("Loading documents...")
        documents = SimpleDirectoryReader(
            directory_path, file_extractor=CUSTOM_FILE_READER_CLS
        ).load_data()

        print("Building index...")
        doc_index = GPTVectorStoreIndex.from_documents(
            documents, service_context=service_context,
            storage_context=storage_context
        )
        doc_index.set_index_id("doc_vector_index")
        keyword_index = None

    print("Persisting index...")
    doc_index.storage_context.persist(persist_dir=persist_dir)

    if mode == "node" and keyword_index is not None:
        keyword_index.storage_context.persist(persist_dir=persist_dir)


@chatbot.command(help="Query index.")
@click.option('--index-path', '-d', required=True, help="The directory which saved the documents.")
@click.option('--index-type', '-i', default="default", help="The type of index.", type=click.Choice(["default", "qdrant", "qdrant-prod"]))
@click.option('--similarity', '-s', default=0.5, help="The similarity threshold.", type=float)
@click.option('--port', '-p', default=7860, help="The port of the server.", type=int)
@click.option('--index-id', '-n', default=None, help="The index id.", type=click.Choice(["doc_vector_index", "keyword_table_index", None]))
@click.option('--similarity-top-k', '-k', default=5, help="The number of similar documents.", type=int)
def query(index_path, index_type, similarity, port, index_id, similarity_top_k):
    if os.path.exists(index_path):
        iface = gr.Interface(fn=launch_chatbot(index_path, index_type=index_type,
                                               similarity=similarity,
                                               index_id=index_id, similarity_top_k=similarity_top_k),
                             inputs=gr.inputs.Textbox(lines=7,
                                                      label="Enter your text"),
                             outputs="text",
                             title="Custom-trained AI Chatbot")

        iface.queue().launch(debug=True, share=False, inline=False, server_port=port)
    else:
        print("Index file not found.")
        return


if __name__ == "__main__":
    chatbot()
