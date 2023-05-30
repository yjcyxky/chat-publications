## Installation and Setup

- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

- Create a conda environment

```
conda create -n llama-index python=3.11.3
conda activate llama-index
```

- Install g++ and CUDA

```
mamba install -c nvidia -c conda-forge cxx-compiler==1.1.3 cuda cuda-toolkit
```

- Install the Python package with `pip install -r requirements.txt`

- Download a [RWKV model](https://huggingface.co/BlinkDL/rwkv-4-raven/tree/main) and place it in your desired directory

    ```
    mkdir models/rwkv-4-raven
    wget xxx -O ./models/rwkv-4-raven/RWKV-4-Raven-14B-v11x-Eng99-Other1-20230501-ctx8192.pth
    wget https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/20B_tokenizer.json -O ./models/rwkv-4-raven/20B_tokenizer.json
    ```

- Download [the tokens file](https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/20B_tokenizer.json)

More details on the model and tokenizer can be found [here](https://python.langchain.com/en/latest/integrations/rwkv.html).
