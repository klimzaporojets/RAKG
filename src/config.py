# Global configuration variables
import os

OLLAMA_BASE_URL = "http://localhost:11434"  # Change this to your Ollama server URL
DEFAULT_MODEL = "qwen2.5:72b"
EMBEDDING_MODEL = "bge-m3:latest"   
SIMILARITY_MODEL = "qwen2:7b"



# OpenAI Configuration
# kzaporoj - commented original
# base_url="https://api.siliconflow.cn/v1" # https://api.siliconflow.cn/v1 for siliconflow
# OPENAI_API_KEY = "your_api_key"  # Set your OpenAI API key here
# OPENAI_MODEL = "Qwen/Qwen2.5-72B-Instruct"  # Default model
# OPENAI_EMBEDDING_MODEL = "BAAI/bge-m3"  # Default embedding model
# OPENAI_SIMILARITY_MODEL = "Qwen/Qwen2.5-14B-Instruct"  # Model for similarity checks

# kzaporoj, added mine:
# OpenAI / proxy configuration
# base_url = 'https://ai-research-proxy.azurewebsites.net'
base_url = 'https://api.openai.com'
base_url_embedding_model = 'https://willma.surf.nl/api/v0'

# base_url = os.getenv(
#     'OPENAI_BASE_URL',
#     'https://ai-research-proxy.azurewebsites.net/v1'
# )

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY_MY')
OPENAI_EMBEDDING_API_KEY = os.getenv('API_KEY_WILLMA_SURF')

OPENAI_MODEL = 'gpt-5.1'
# OPENAI_MODEL = 'nf-gpt-4o-mini'
# OPENAI_EMBEDDING_MODEL = 'text-embedding-3-large'
# OPENAI_EMBEDDING_MODEL = 'nf-text-embedding-ada-002'
OPENAI_EMBEDDING_MODEL = 'Qwen/Qwen3-Embedding-8B'
# OPENAI_SIMILARITY_MODEL = 'nf-gpt-4o-mini'
OPENAI_SIMILARITY_MODEL = 'gpt-5.1'


# Model Provider Selection
USE_OPENAI = True  # Set to True to use OpenAI, False to use Ollama