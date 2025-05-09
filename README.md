## 📘 PDF ChatBot

A Streamlit application leveraging Retrieval-Augmented Generation (RAG) to answer user queries about PDF documents. It integrates OpenAI embeddings, FAISS vector store, and DeepEval evaluation for robust, context-aware responses.

---

### 🗂️ Repository Structure

```
sharikjavid-pdf_chat/
├── README.md               # Project overview and setup instructions
├── app.py                  # Streamlit entry point
├── config.py               # Application configuration and validation
├── eval.py                 # DeepEval-based evaluation script
├── requirements.txt        # Python dependencies
├── vector_loader.py        # injection pipline
├── rag_components/         # RAG pipeline components
│   ├── document_parser.py  # Parse raw docs into text/images
│   ├── prompt_builder.py   # Construct LLM prompts
│   ├── resource_loader.py  # Load embeddings, vector store, LLM
│   ├── retrieval_chain.py  # Build and invoke RAG chain
│   └── prompt/             # Prompt templates
│       └── prompts.py      # Generation and reasoning templates
├── ui/                     # Streamlit UI components
│   └── main_ui.py          # Chat interface and sidebar display
├── vectorstore/            # Persisted vector database
│   ├── db_faiss/           # FAISS index files
│   └── docstore.pkl        # Document store
└── .deepeval/              # DeepEval cache and telemetry
```

---

### ⚙️ Setup & Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/sharikjavid/sharikjavid-pdf_chat.git
   cd sharikjavid-pdf_chat
   ```

2. **Configure environment**

   * Create a `.env` file in the project root with the following content:

     ```ini
     OPENAI_API_KEY=your_openai_api_key_here
     ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

5. **Evaluate performance**

   ```bash
   python eval.py
   ```
