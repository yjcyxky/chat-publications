# Chat Publications
Answer research questions based on publications from PubMed.

## Introduction

You may have so many research questions about specific topics, such as:
- What's the Long Covid? 
- Any papers mentioned relationships between TP53 gene and lung cancer?
- More questions ...

You may find the answers from PubMed, but it is not easy to find the answers from so many publications. By using the state-of-the-art language models (LLM), such as [Vicuna](https://vicuna.lmsys.org/) and [ChatGPT](https://chat.openai.com/), you can get the answers from publications easily.

But how to connect the LLM's (like Vicuna, ChatGPT) with publications from pubmed? This project provides a solution to connect the LLM's with publications from pubmed. It's based on LLM (Vicuna or ChatGPT), [LLAMA index](https://github.com/jerryjliu/llama_index), 
[LangChain](https://github.com/hwchase17/langchain), [FastChat](https://github.com/lm-sys/FastChat) etc.

- Do you want to connect your LLM's (like Vicuna, ChatGPT) with your own external data (such as publications from pubmed). You also can follow the [python script](https://github.com/yjcyxky/chat-publications/blob/main/chatbot_vicuna.py) as an example to build your own chatbot server.

- If you want to know more details about LLM and LLAMA index, please access [more resources](./resources.md)

## Download all publication data (not full text, just title, abstract, authors, etc.) from PubMed

Please access [here](./pubmed/README.md) for more details.

## How to run vicuna with llama-index?

### Clone the current repository

```
git clone https://github.com/yjcyxky/chat-publications.git
cd chat-publications
```

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

### Launch all services by systemd

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

### Launch all services step by step manually
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

### Build index for my own data

CAUTION: It assumes that your data is in `${PWD}/data/my-project` directory.

```
# Please use custom-http option to connect the above server (OpenAI's API, we assume that it is running on localhost:8000, If not, please change chatbot_vicuna.py file)
python3 chatbot_vicuna.py index -d data/my-project -l custom-http

# If you changed your data, you need to rebuild the index
```

### Launch chatbot server

After this step, you will get your own chatbot server at http://localhost:7860

```
# By systemd
systemctl start chatbot.service

# Manually
/data/miniconda3/envs/llama-index/bin/python3 chatbot_vicuna.py query -d data/my-project -l custom-http
```

### [Optional] Proxy chatbot server with nginx

```
cp nginx/chatbot.conf /etc/nginx/conf.d/

# Modify the chatbot.conf file to change the server_name and proxy_pass based on your own environment

systemctl restart nginx
```

## How to run ChatGPT with llama-index?
### How to build index for my own data?

```
python chatbot.py index -d data/my-project
```

### Examples

1. What's the Long Covid? which papers mentioned it? Could you list all doi numbers of these papers?

```
# Vicuna
Answer: The Long Covid is a term used in several of the papers listed in the context information to refer to the long-term effects and persistence of COVID-19 symptoms and illness after acute infection. The papers that mention the Long Covid are:

1. with in/f\_lammation pro/f\_iling of \nlong-COVID and association with one-year recovery following \nhospitalisation in the UK: a prospective observational study. Preprint at \nmedRxiv <https://doi.org/10.1101/2021.12.13.21267471> (2021).
2. Taquet, M. et al. Incidence, co-occurrence, and evolution of long-COVID \nfeatures: a 6-month retrospective cohort study of 273,618 survivors of \nCOVID-19. PLoS Med. 18, e1003773 (2021).
3. Estiri, H. et al. Evolving phenotypes of non-hospitalized patients that \nindicate long COVID. BMC Med. 19, 249 (2021).
4. Caspersen, I. H., Magnus, P. & Trogstad, L. Excess risk and clusters of \nsymptoms a/f_ter COVID-19 in a large Norwegian cohort. Eur. J. Epidemiol. <https://doi.org/10.1007/s10654-022-00847-8> (2022).
5. Havervall, S. et al. Symptoms and functional impairment assessed 8 months \na/f_ter mild COVID-19 among health care workers. J. Am. Med. Assoc. 325, 2015\u20132016 (2021).
6. Blomberg, B. et al. Long COVID in a prospective cohort of home-isolated \npatients. Nat. Med. 27, 1607\u20131613 (2021).
7. Logue, J. K. et al. Sequelae in adults at 6 Months a/f_ter COVID-19 infection. \nJAMA Netw. Open 4, e210830 (2021).
8. Amin-Chowdhury, Z. et al. Characterising long COVID more than 6 \nmonths a/f_ter acute infection in adults; prospective longitudinal cohort
```

2. Is ME/CFS a cause of fatigue?

```
# Vicuna
Yes, ME/CFS is a cause of fatigue.

No, ME/CFS is not a cause of fatigue.

I'm not sure.
```

3. Which relationship is between Fatigue and ME/CFS?

```
# Vicuna
A. Fatigue is a symptom of ME/CFS
B. ME/CFS is a cause of fatigue
C. Fatigue is a cause of ME/CFS
D. There is no relationship between fatigue and ME/CFS

What is the answer?
```

4. Which relationship is between Fatigue and Long Covid?

```
# Vicuna
A) Fatigue is a symptom of Long Covid
B) Long Covid is a cause of Fatigue
C) Fatigue is a symptom of Pathological Fatigue
D) Long Covid is a cause of Pathological Fatigue
E) Fatigue is a symptom of Pathological Fatigue and Long Covid is a cause of Pathological Fatigue

Answer: B) Long Covid is a cause of Fatigue

Explanation:
The text states that Long Covid can cause fatigue and that this fatigue is different from physiological fatigue, which is easily cured by rest. It also mentions that pathological fatigue may be caused by factors such as viral or bacterial infection, trauma, disease, or other cellular assault, and that the cellular metabolism changes do not always reset after providing energy for the defense/repair of the body. Therefore, Long Covid is a cause of fatigue, but it is not necessarily a cause of pathological fatigue.
```
