### Make an environment for llama-index and vicuna

```bash
conda create -n llama-index python=3.11.3
conda activate llama-index

# If you have any problem at this step, please post an issue
pip3 install -r requirements.txt
```


### Download the LLama model

Get the llama-13b model weights from decapoda-research/llama-13b-hf

```
python get_llama.py

# ValueError: Tokenizer class LLaMATokenizer does not exist or is not currently imported.
# Avoid to use AutoTokenizer and AutoModelForCausalLM
```

### Apply the vicuna-13b-delta weight on the LLama model

CAUTION: It assumes that 
1. you have got the llama-13b model weights in `/root/.cache/huggingface/hub/models--decapoda-research--llama-13b-hf/snapshots/438770a656712a5072229b62256521845d4de5ce` directory.
2. the output directory is `/data/vicuna-13b` directory.

```
python3 -m fastchat.model.apply_delta \
    --base-model-path /root/.cache/huggingface/hub/models--decapoda-research--llama-13b-hf/snapshots/438770a656712a5072229b62256521845d4de5ce \
    --target-model-path /data/vicuna-13b \
    --delta-path lmsys/vicuna-13b-delta-v1.1
```

### Build index for my own data

CAUTION: It assumes that your data is in `${PWD}/data/my-project` directory.

```
# Please use custom-http option to connect the above server (OpenAI's API, we assume that it is running on localhost:8000, If not, please change chatbot.py file)
python3 chatbot.py index -d data/my-project -l custom-http

# If you changed your data, you need to rebuild the index
```

### Launch chatbot server

After this step, you will get your own chatbot server at http://localhost:7860

```
# By systemd
# You need to modify the chatbot.service file to change the `WorkingDirectory` and `data/my-project` based on your environment.
systemctl start chatbot.service

# Manually
/data/miniconda3/envs/llama-index/bin/python3 chatbot.py query -d data/my-project -l custom-http
```

### Launch all services by systemd for production

#### Copy all files in systemd directory to /etc/systemd/system/ and start all services

CAUTION: It assumes that 
1. You have installed all dependencies into `/data/miniconda3/envs/llama-index` directory.
2. You have vicuna-13b model in `/data/vicuna-13b` directory.

```bash
cp -r systemd/* /etc/systemd/system/

systemctl daemon-reload

# Launch the controller
systemctl start vicuna-controller.service

# Launch the model worker(s)
# The following step need to wait for a while, you can check the status by `journalctl -u vicuna-worker.service`
# If you can see "Uvicorn running on ...", it means that the model worker is ready.
systemctl start vicuna-worker.service

# Launch the openai compatible server
systemctl start vicuna-openai.service

# How to check the status of all services?
python3 test_vicuna_openai_api.py
```

### Launch all services step by step manually for development
#### Run fschat

To serve using the web UI, you need three main components: web servers that interface with users, model workers that host one or more models, and a controller to coordinate the webserver and model workers. You can learn more about the architecture here.
Here are the commands to follow in your terminal:

- Launch the controller

```
python3 -m fastchat.serve.controller
```

This controller manages the distributed workers.

- Launch the model worker(s)

```
python3 -m fastchat.serve.model_worker --model-path /data/vicuna-13b
```

Wait until the process finishes loading the model and you see "Uvicorn running on ...". The model worker will register itself to the controller .

- Launch the Gradio web server

```
python3 -m fastchat.serve.gradio_web_server
```

This is the user interface that users will interact with.

By following these steps, you will be able to serve your models using the web UI. You can open your browser and chat with a model now. If the models do not show up, try to reboot the gradio web server.

- Launch the server which is compatible with OpenAI's API

```
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000 http://localhost:21001
```

### [Optional] Proxy chatbot server with nginx

```
cp nginx/chatbot.conf /etc/nginx/conf.d/

# Modify the chatbot.conf file to change the server_name and proxy_pass based on your own environment

systemctl restart nginx
```