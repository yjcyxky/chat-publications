#!/bin/bash

/data/miniconda3/envs/llama-index/bin/python3 -m fastchat.serve.controller
/data/miniconda3/envs/llama-index/bin/python3 -m fastchat.serve.cli --model-path /data/vicuna-13b
/data/miniconda3/envs/llama-index/bin/python3 -m fastchat.serve.gradio_web_server
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000