
import os
import torch 
from huggingface_hub import hf_hub_download
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import config

if config.llm_model_id == 'llama-3.2-1b' or config.cfg['vram'] == 'no_vram':
    REPO_ID = 'bartowski/Llama-3.2-1B-Instruct-GGUF'
    hf_hub_download(repo_id=REPO_ID, filename='Llama-3.2-1B-Instruct-Q4_K_M.gguf', local_dir='models/Llama-3.2-1B-Instruct-Q4_K_M.gguf')
elif config.llm_model_id ==  'llama-3.2-3b' or config.cfg['vram'] == 'low_vram':
    REPO_ID = 'bartowski/Llama-3.2-3B-Instruct-GGUF'
    hf_hub_download(repo_id=REPO_ID, filename='Llama-3.2-3B-Instruct-Q4_K_M.gguf', local_dir='models/Llama-3.2-3B-Instruct-Q4_K_M.gguf')
elif config.llm_model_id ==  'llama-3.2-8b' or config.cfg['vram'] == 'high_vram':
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm = HuggingFaceLLM(
            model_name=model_name,
            tokenizer_name=model_name,
            device_map="auto",
            model_kwargs={"torch_dtype": torch.bfloat16, "cache_dir":config.models_dir},
            # generate_kwargs={"pad_token_id": tokenizer.eos_token_id},
            tokenizer_kwargs={"padding_side": "left", "cache_dir":config.models_dir},
            context_window=2048,
            max_new_tokens=1024,
        )


# Downlaod Embedding models 
print(config.embed_model)
HuggingFaceEmbedding(model_name=config.embed_model, cache_folder=config.models_dir)
