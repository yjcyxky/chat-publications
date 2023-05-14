### How to build index for my own data?

```
python chatbot.py index -d docs/type2-dm
```

### Examples

1. What's the Long Covid? which papers mentioned it? Could you list all doi numbers of these papers?

```
# Dolly
The Long Covid was first mentioned in paper no. 2021, which is titled Incidence, co-occurrence, and evolution of long-COVID features: a 6-month retrospective cohort study of 273,618 survivors of COVID-19. The Long Covid is defined as the symptoms lasting more than 6 months after infection with COVID-19. The papers no. 2021, 2022, 2023, 2024, and 2025 mention the Long Covid and have the doi numbers 10.1007/s10654-022-00847-8, 2015–2016,  J. Am. Med. Assoc.  325, 2015–2016,  and Eur. J. Epidemiol.  https://doi.org/10.1007/s10654-022-00847-8, respectively.

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

### How to run vicuna?

#### Download the LLama model

```
python get_llama.py

# ValueError: Tokenizer class LLaMATokenizer does not exist or is not currently imported.
# Avoid to use AutoTokenizer and AutoModelForCausalLM
```

#### Install fschat

```
pip3 install fschat
```

### Apply the vicuna-13b-delta weight on the LLama model

```
python3 -m fastchat.model.apply_delta \
    --base-model-path /root/.cache/huggingface/hub/models--decapoda-research--llama-13b-hf/snapshots/438770a656712a5072229b62256521845d4de5ce \
    --target-model-path /root/vicuna-13b \
    --delta-path lmsys/vicuna-13b-delta-v1.1
```

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
python3 -m fastchat.serve.model_worker --model-path /path/to/model/weights
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
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```

### Replace the openai.py file in the llama-index library

```
cp embeddings/openai.py /xxx/site-packages/llama_index/embeddings/openai.py
```