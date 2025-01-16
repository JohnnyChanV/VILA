# from vllm import LLM, SamplingParams
import time

from tqdm import tqdm
import pandas as pd
from googletranslatepy import Translator
import json
import concurrent.futures
import os

os.environ['HuggingfaceCache'] = "/root/data/model_cache"
os.environ['https_proxy'] = "http://127.0.0.1:7890"
os.environ['http_proxy'] = "http://127.0.0.1:7890"

cache_dir = os.environ['HuggingfaceCache']
access_token = "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"
device = "cuda"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"



df = pd.read_csv('data/advbench.csv')[:10]
hindi_prompts = df['goal_hi'].to_list()


sampling_params = SamplingParams(temperature=0.9,
                                 seed=42,
                                 max_tokens=256)

llm = LLM(model = MODEL_NAME ,
          token=access_token,
          trust_remote_code=True,
            cache_dir=cache_dir)

outputs = llm.generate(hindi_prompts,sampling_params)

responses = []
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    responses += generated_text

df['response_hindi'] = responses

df.to_csv("data/hindi_advbench_results.csv",encoding='utf-8')

