from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

from src.config import (
    OLLAMA_BASE_URL, DEFAULT_MODEL, EMBEDDING_MODEL, SIMILARITY_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL, OPENAI_EMBEDDING_MODEL, OPENAI_SIMILARITY_MODEL,
    USE_OPENAI, OPENAI_EMBEDDING_API_KEY, base_url, base_url_embedding_model
)
from openai import OpenAI


class EmbeddingModelAdapter:
    def __init__(self, *, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def embed_documents(self, texts):
        resp = self.client.embeddings.create(model=self.model, input=texts)
        data = sorted(resp.data, key=lambda d: d.index)  # <-- critical
        return [d.embedding for d in data]

    def embed_query(self, text):
        resp = self.client.embeddings.create(model=self.model, input=[text])
        assert len(resp.data) == 1
        return resp.data[0].embedding


class LLMProvider:
    def __init__(self):
        if USE_OPENAI:
            # 使用 ChatOpenAI 并启用 JSON 模式
            print(f'self.llm_created_chatopenai: OPENAI_MODEL {OPENAI_MODEL}, '
                  f'OPENAI_API_KEY {OPENAI_API_KEY} '
                  f'base_url {base_url} ')
            self.llm = ChatOpenAI(
                model=OPENAI_MODEL,
                api_key=OPENAI_API_KEY,
                base_url=base_url,
                temperature=0,
            ).bind(response_format={"type": "json_object"})

            # self.embedding_model = OpenAIEmbeddings(
            #     model=OPENAI_EMBEDDING_MODEL,
            #     # api_key=OPENAI_API_KEY,
            #     api_key=OPENAI_EMBEDDING_API_KEY,
            #     # base_url=base_url
            #     base_url=base_url_embedding_model,
            #     tiktoken_enabled=False,
            # )
            self.embedding_model = EmbeddingModelAdapter(
                model=OPENAI_EMBEDDING_MODEL,
                api_key=OPENAI_EMBEDDING_API_KEY,
                base_url=base_url_embedding_model,  # e.g. https://willma.surf.nl/api/v0
            )
            # self.embedding_model = OpenAI(
            #     api_key=OPENAI_EMBEDDING_API_KEY,
            #     base_url=base_url_embedding_model,  # e.g. https://willma.surf.nl/api/v0
            # )
            self.similarity_model = ChatOpenAI(
                model=OPENAI_SIMILARITY_MODEL,
                api_key=OPENAI_API_KEY,
                base_url=base_url,
                temperature=0,
            ).bind(response_format={"type": "json_object"})
        else:
            # 使用 Ollama 模型
            self.llm = OllamaLLM(
                model=DEFAULT_MODEL,
                base_url=OLLAMA_BASE_URL,
                format='json',
                temperature=0
            )
            self.embedding_model = OllamaEmbeddings(
                model=EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL
            )
            self.similarity_model = OllamaLLM(
                model=SIMILARITY_MODEL,
                base_url=OLLAMA_BASE_URL,
                format='json',
                temperature=0
            )

    def get_llm(self):
        return self.llm

    def get_embedding_model(self):
        return self.embedding_model

    def get_similarity_model(self):
        return self.similarity_model
