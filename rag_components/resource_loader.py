import pickle
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore # Or other stores like LocalFileStore

from config import AppConfig 

class ResourceLoader:
    def __init__(self, config: AppConfig):
        self.config = config
        self.embedding_model = None
        self.vectorstore = None
        self.docstore = None
        self.retriever = None
        self.llm = None

    def load_all(self):
        """Loads all necessary resources."""
        print("[INFO] ResourceLoader: Validating configuration...")
        self.config.validate_config() # Validate paths and API key presence

        print("[INFO] ResourceLoader: Loading embedding model...")
        self.embedding_model = OpenAIEmbeddings(openai_api_key=self.config.OPENAI_API_KEY)
        
        print(f"[INFO] ResourceLoader: Loading vector store from {self.config.DB_FAISS_PATH}...")
        self.vectorstore = FAISS.load_local(
            self.config.DB_FAISS_PATH, 
            self.embedding_model, 
            allow_dangerous_deserialization=True
        )
        
        print(f"[INFO] ResourceLoader: Loading document store from {self.config.DOCSTORE_PATH}...")
        with open(self.config.DOCSTORE_PATH, "rb") as f:
            self.docstore = pickle.load(f)
            if not isinstance(self.docstore, InMemoryStore):
                 print(f"[WARN] Loaded docstore is of type {type(self.docstore)}, not InMemoryStore. Ensure compatibility.")


        print("[INFO] ResourceLoader: Initializing MultiVectorRetriever...")
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key="doc_id", 
        )
        self.retriever.search_kwargs = self.config.RETRIEVER_SEARCH_KWARGS
        
        print(f"[INFO] ResourceLoader: Initializing LLM ({self.config.LLM_MODEL_NAME})...")
        self.llm = ChatOpenAI(
            model=self.config.LLM_MODEL_NAME, 
            temperature=0, 
            openai_api_key=self.config.OPENAI_API_KEY
        )
        print("[INFO] ResourceLoader: All resources loaded successfully.")

    def get_retriever(self):
        if not self.retriever:
            self.load_all()
        return self.retriever

    def get_llm(self):
        if not self.llm:
            self.load_all()
        return self.llm