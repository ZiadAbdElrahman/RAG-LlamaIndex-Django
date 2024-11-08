import os
import torch
import time

from llama_index.core import Settings
from llama_index.core import load_index_from_storage
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.llama_cpp import LlamaCPP

from pdf_preprocessor import get_pdf_documents, get_pdf_documents_with_context
from utils import hf_autotokenizer_to_chat_formatter

from config import config
from custom_logger import get_logger

logger = get_logger()

model_id2model_name = {
    'llama-3.2-1b':'Llama-3.2-1B-Instruct-Q4_K_M.gguf',
    'llama-3.2-3b':'Llama-3.2-3B-Instruct-Q4_K_M.gguf',
    'llama-3.2-8b':'meta-llama/Meta-Llama-3-8B-Instruct'
}




class RAG:
    def __init__(self):
        gpu_available = os.getenv('USING_GPU', 0)
        if not gpu_available:
            config.vram = 'no_vram'
            
        logger.info(f'loading {config.llm_model_id} model, running on {config.vram} mode, using {config.device}')
        self.llm = self.initialize_llm()
        logger.info('LLM model is Loaded')
        
        Settings.embed_model = HuggingFaceEmbedding(model_name=config.embed_model, cache_folder=config.models_dir)
        logger.info('Embedding model is Loaded')

        Settings.text_splitter = SentenceSplitter(chunk_size=1500)
        self.active_chatbots = {}
        self.chatbot_last_use = {}
        
    def add_pdf_file(self, user_id, pdf_dir):
        documents = get_pdf_documents(pdf_dir)
        # documents = get_pdf_documents_with_context(pdf_dir, self.llm._model)
        vector_store = MilvusVectorStore(
            uri = os.path.join(config.vector_db_dir, user_id+'.dp'),
            dim = 768
            )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        logger.info(f'file {pdf_dir} is added for {user_id}')
        
    def start_chat(self, user_id):
        vector_store = MilvusVectorStore(
            uri = os.path.join(config.vector_db_dir, user_id+'.dp'),
            dim = 768
            )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
        self.active_chatbots[user_id] = index.as_chat_engine(chat_mode="condense_plus_context", llm=self.llm, similarity_top_k=20)
        self.chatbot_last_use[user_id] = time.time()
    
    def get_respond(self, user_id, user_message):
        if not user_id in list(self.active_chatbots.keys()):
            self.start_chat(user_id)
        
        self.chatbot_last_use[user_id] = time.time()
        respond = str(self.active_chatbots[user_id].chat(user_message))
        respond = respond.replace('assistant\n\n', '')
        return respond
        
    def deactivate_idle_chatbots(self):
        for chat_id in self.active_chatbots.keys():
            if (time.time() - chatbot_last_use[chat_id]) // 60 > 5: #if chatbot idle for 5 minutes it will be deactivated
                del self.active_chatbots[chat_id]
                del self.chatbot_last_use[chat_id]
        
    def initialize_llm(self):
        if '1b' in config.llm_model_id or '3b' in config.llm_model_id:
            chat_formatter = hf_autotokenizer_to_chat_formatter('meta-llama/Llama-3.2-3B-Instruct')
            
            llm = LlamaCPP(
                model_path=f'{config.models_dir}/{model_id2model_name[config.llm_model_id]}/{model_id2model_name[config.llm_model_id]}',
                temperature=0.5,
                max_new_tokens=1024,
                context_window=8192,
                generate_kwargs={},
                model_kwargs={} if config.device == 'cpu' else {"n_gpu_layers": -1},
                messages_to_prompt=chat_formatter,
                verbose=False,
            )
        elif '8b' in config.llm_model_id:
            llm = HuggingFaceLLM(
                model_name=model_id2model_name[config.llm_model_id],
                tokenizer_name=model_id2model_name[config.llm_model_id],
                device_map=config.device,
                model_kwargs={"torch_dtype": torch.bfloat16, "cache_dir":config.models_dir},
                # generate_kwargs={"pad_token_id": tokenizer.eos_token_id},
                tokenizer_kwargs={"padding_side": "left", "cache_dir":config.models_dir},
                context_window=32768,
                max_new_tokens=2048,
            )
        return llm

if __name__ == '__main__':
    rag = RAG()
    rag.start_chat('testing_id')
    print(rag.get_respond('ziii', ' what information do you know?'))
    print(rag.get_respond('ziii', 'how much experience does he has?'))