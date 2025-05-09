import os
from dotenv import load_dotenv
load_dotenv()



class AppConfig:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
    DB_FAISS_PATH = 'vectorstore/db_faiss'
    DOCSTORE_PATH = 'vectorstore/docstore.pkl'
    LLM_MODEL_NAME = "gpt-4o-mini"
    RETRIEVER_SEARCH_KWARGS = {"k": 10} 



    @staticmethod
    def validate_config():
        if not AppConfig.OPENAI_API_KEY or AppConfig.OPENAI_API_KEY == "YOUR_FALLBACK_OPENAI_KEY_IF_NOT_IN_ENV":
            raise ValueError("OpenAI API key not found. Please set it in the .env file or as an environment variable.")
        if not os.path.exists(AppConfig.DB_FAISS_PATH):
            raise FileNotFoundError(f"FAISS database not found at {AppConfig.DB_FAISS_PATH}")
        if not os.path.exists(AppConfig.DOCSTORE_PATH):
            raise FileNotFoundError(f"Document store not found at {AppConfig.DOCSTORE_PATH}")

