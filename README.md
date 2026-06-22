# RAG Complaint Chatbot

A Retrieval-Augmented Generation (RAG) powered chatbot that analyzes 
customer complaints for CrediTrust Financial.

## Project Structure
- `notebooks/` - EDA and preprocessing notebooks
- `src/` - Core RAG pipeline source code
- `vector_store/` - Persisted FAISS/ChromaDB index
- `data/` - Raw and processed complaint data
- `app.py` - Gradio chat interface

## Setup
```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

## Tasks
- Task 1: EDA and Data Preprocessing
- Task 2: Text Chunking, Embedding and Vector Store
- Task 3: RAG Core Logic and Evaluation
- Task 4: Interactive Chat Interface