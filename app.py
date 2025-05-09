import streamlit as st 
st.set_page_config(page_title="PDF Chatbot", layout="wide", initial_sidebar_state="auto")

from config import AppConfig
from rag_components.resource_loader import ResourceLoader
from rag_components.retrieval_chain import RAGChainManager
from ui.main_ui import MainUI


@st.cache_resource 
def get_rag_chain_manager() -> RAGChainManager:

    try:
        print("[INFO] app.py: Attempting to load resources and initialize RAGChainManager...")
        config = AppConfig() 
        resource_loader = ResourceLoader(config)
        resource_loader.load_all()
        
        rag_manager = RAGChainManager(resource_loader)
        print("[INFO] app.py: RAGChainManager initialized successfully.")
        return rag_manager
    except (FileNotFoundError, ValueError) as e: 
        print(f"[ERROR] app.py: Initialization failed in get_rag_chain_manager: {e}")
        raise 
    except Exception as e:
        print(f"[ERROR] app.py: Unexpected initialization error in get_rag_chain_manager: {e}")
        import traceback
        print(traceback.format_exc())
        raise

def main():
    """
    Main function to run the Streamlit application.
    """
    rag_manager = None
    try:
        rag_manager = get_rag_chain_manager()
    except (FileNotFoundError, ValueError) as e:
        st.error(f"Initialization Error: {e}") # 'st' is available here
        print(f"[ERROR] app.py (main): Initialization failed: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during initialization: {e}") # 'st' is available here
        print(f"[ERROR] app.py (main): Unexpected initialization error: {e}")

    if rag_manager:
        ui = MainUI(rag_manager)
        ui.run()
    else:
        print("[ERROR] app.py (main): RAG Manager is None, UI cannot start. An error should have been displayed on the page.")


if __name__ == "__main__":
    main()