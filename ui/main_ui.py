import streamlit as st
from typing import Optional, List, Dict

from rag_components.retrieval_chain import RAGChainManager 


class MainUI:
    def __init__(self, rag_chain_manager: RAGChainManager):
        self.rag_chain_manager = rag_chain_manager
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initializes session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "retrieved_texts_for_display" not in st.session_state:
            st.session_state.retrieved_texts_for_display = []
        if "retrieved_images_for_display" not in st.session_state:
            st.session_state.retrieved_images_for_display = []


    def display_chat_history(self):
        """Displays the chat history."""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def display_retrieved_content_sidebar(self):
        """Displays retrieved text and images in the sidebar."""
        with st.sidebar:
            st.subheader("📄 Retrieved Context (for last query)")
            
            if not st.session_state.retrieved_texts_for_display and not st.session_state.retrieved_images_for_display:
                 st.info("No content retrieved yet, or retrieval was empty.")

            if st.session_state.retrieved_texts_for_display:
                st.markdown("**Texts:**")
                for i, doc_content in enumerate(st.session_state.retrieved_texts_for_display):
                    st.text_area(f"Doc {i+1}", doc_content[:10000], height=150, key=f"ret_text_{i}")


            if st.session_state.retrieved_images_for_display:
                st.markdown("**Images:**")
                for i, img_b64 in enumerate(st.session_state.retrieved_images_for_display):
                     st.image(f"data:image/jpeg;base64,{img_b64}", caption=f"Image {i+1}")


    def run(self):
        """Runs the Streamlit UI application."""
        st.title("📘 Pdf ChatBot")

        self.display_chat_history()

        self.display_retrieved_content_sidebar() 

        if prompt := st.chat_input("Ask a question about the document"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                       
                        parsed_docs_for_sidebar = self.rag_chain_manager.retrieve_documents(prompt)
                        
                        st.session_state.retrieved_texts_for_display = [
                            doc.page_content for doc in parsed_docs_for_sidebar.get("texts", [])
                        ]
                        st.session_state.retrieved_images_for_display = parsed_docs_for_sidebar.get("images", [])
                        
            
                        response = self.rag_chain_manager.invoke(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        st.rerun()

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
