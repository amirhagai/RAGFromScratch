from load_env_keys import load_env_keys, offline_transformers

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

def get_llama(local_model_path="meta-llama/Llama-3.2-3B-Instruct", max_new_tokens=2048):

    # Load the tokenizer and model from the local directory
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)

    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",

    )
    model.config.pad_token_id = tokenizer.eos_token_id  = None

    # Create a text-generation pipeline with the loaded model and tokenizer
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=max_new_tokens,
        do_sample=False, 
        temperature=None,
        top_p=None,
    )

    # Integrate the pipeline with LangChain's HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return llm





