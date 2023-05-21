## Installation and Setup

- Install the Python package with `pip install rwkv`
- Install the tokenizer Python package with `pip install tokenizer`
- Download a [RWKV model](https://huggingface.co/BlinkDL/rwkv-4-raven/tree/main) and place it in your desired directory

    ```
    wget xxx -O /data/rwkv-14b-v11/RWKV-4-Raven-14B-v11x-Eng99-Other1-20230501-ctx8192.pt
    wget https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/20B_tokenizer.json -O /data/rwkv-14b-v11/20B_tokenizer.json
    ```

- Download [the tokens file](https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/20B_tokenizer.json)

More details on the model and tokenizer can be found [here](https://python.langchain.com/en/latest/integrations/rwkv.html).