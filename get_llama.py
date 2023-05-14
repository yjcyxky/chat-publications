#!/usr/bin/env python3

from transformers import LlamaForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-13b-hf")

mode = LlamaForCausalLM.from_pretrained("decapoda-research/llama-13b-hf")
