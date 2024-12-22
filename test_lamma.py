from load_env_keys import load_env_keys, offline_transformers
load_env_keys()
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
from langchain_huggingface import HuggingFacePipeline


# Set random seeds for reproducibility
set_seed(42)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# Ensure deterministic algorithms in PyTorch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define the path to your local model directory
local_model_path = "meta-llama/Llama-3.2-3B-Instruct"

# Load the tokenizer and model from the local directory
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Create a text-generation pipeline with the loaded model and tokenizer
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=2048,
    do_sample=False, 
)

# Integrate the pipeline with LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

print(llm.invoke("do you think it will rain today?"))
