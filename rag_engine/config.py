import os
from pathlib import Path
from typing import TypedDict

import yaml

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub.hf_api import HfFolder 

class Singleton(type):
    _instances = {}  # type: ignore

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

vram2embed_model = {
    'no_vram':'BAAI/bge-base-en-v1.5',
    'low_vram':'BAAI/bge-large-en-v1.5',
    'high_vram':'BAAI/bge-reranker-large'
}

embed_model2dim = {
    'BAAI/bge-base-en-v1.5':768,
    'BAAI/bge-large-en-v1.5':1024,
    'BAAI/bge-reranker-large':1024
}


class AppConfig(metaclass=Singleton):  # pylint: disable=too-many-instance-attributes
    def __init__(self, config_file):
        self.cfg = self._load_config_file(config_file)

        self.vram = self.cfg['vram'] # high_vram, low_vram, no_vram
        self.vector_db_dir = self.cfg['vector_db_dir']
        self.models_dir = self.cfg['models_dir']
        self.llm_model_id = self.cfg['llm_model_id']
        self.use_vlm = self.cfg['use_vlm']
        self.table_extraction = self.cfg['table_extraction']

        gpu_available = os.getenv('USING_GPU', -1)
        if gpu_available==0:
           self.vram = 'no_vram'

        if self.llm_model_id == 'none':
            if self.vram == 'no_vram':
                self.llm_model_id = 'llama-3.2-1b'
            elif self.vram == 'low_vram':
                self.llm_model_id = 'llama-3.2-3b'
            elif self.vram == 'high_vram':
                self.llm_model_id = 'llama-3.2-8b'
                
        self.embed_model = vram2embed_model[self.vram]
        self.embed_model_dim = embed_model2dim[self.embed_model]
        self.device = 'cpu' if self.vram == 'no_vram' else 'cuda'
        
    @staticmethod
    def _load_config_file(config_file):
        if not os.path.exists(config_file):
            raise Exception(f'Config file {config_file} not found')
        with open(config_file, 'r', encoding='UTF-8') as stream:
            cfg = yaml.safe_load(stream)
            for key in cfg:
                env_var_name = key.upper()
                if env_var_name in os.environ:
                    print('Overriding config value for', key, 'with value from env var', env_var_name)  # noqa: T201
                    cfg[key] = os.environ[env_var_name]

            return cfg




ENV = os.getenv('ENV', 'dev')

config_file_path = os.path.join(Path(__file__).parent.resolve(), 'config', f'{ENV.lower()}.yaml')
# print('Using config file:', config_file_path)  # noqa: T201
config = AppConfig(config_file_path)

os.environ['HF_HOME'] = config.cfg['models_dir']

HfFolder.save_token(os.getenv('HUGGING_FACE_HUB_TOKEN', 'None'))
