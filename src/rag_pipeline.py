# ============================================================
# Task 3: RAG Core Logic
# CrediTrust Financial - RAG Complaint Chatbot
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from transformers import pipeline as hf_pipeline

BASE_DIR = r'C:\Users\HP\Desktop\my files\rag-complaint-chatbot'
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store", "complaints.index")
METADATA_PATH = os.path.join(BASE_DIR, "vector_store", "chunks_metadata.csv")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vector_store", "vectorizer.pkl")
TOP_K = 5

# ── LOAD RESOURCES ──────────────────────────────────────────
print("Loading vector store...")
index = faiss.read_index(VECTOR_STORE_PATH)
chunks_df = pd.read_csv(METADATA_PATH)

print("Loading vectorizer...")
with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer, svd = pickle.load(f)

print(f"✅ RAG pipeline ready! Index has {index.ntotal} vectors.")


# ── RETRIEVER ───────────────────────────────────────────────
def retrieve(query: str, k: int = TOP_K):
    query_tfidf = vectorizer.transform([query])
    query_embedding = svd.transform(query_tfidf).astype(np.float32)
    distances, indices = index.search(query_embedding, k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and idx < len(chunks_df):
            row = chunks_df.iloc[idx]
            results.append({
                'text': str(row['chunk_text']),
                'product_category': str(row.get('product_category', 'Unknown')),
                'complaint_id': str(row.get('complaint_id', 'N/A')),
                'issue': str(row.get('issue', 'N/A')),
                'distance': float(distances[0][i])
            })
    return results


# ── PROMPT BUILDER ──────────────────────────────────────────
def build_prompt(question: str, chunks: list) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i} - {chunk['product_category']}]\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)
    return f"""You are a financial analyst assistant for CrediTrust Financial.
Your task is to answer questions about customer complaints.
Use ONLY the following retrieved complaint excerpts to formulate your answer.
If the context does not contain enough information, say so clearly.
Be concise and helpful. Answer in 3-5 sentences.

Context:
{context}

Question: {question}

Answer:"""


# ── LOAD LLM ────────────────────────────────────────────────
def load_llm():
    print("Loading LLM (gpt2)...")
    llm = hf_pipeline(
        "text-generation",
        model="gpt2",
        max_new_tokens=200,
    )
    print("✅ LLM loaded!")
    return llm


def generate_answer(prompt: str, llm) -> str:
    result = llm(prompt, max_new_tokens=150, do_sample=False, truncation=True)
    if isinstance(result, list):
        generated = result[0].get('generated_text', '').strip()
        if "Answer:" in generated:
            return generated.split("Answer:")[-1].strip()
        return generated
    return str(result).strip()

# ── FULL RAG PIPELINE ───────────────────────────────────────
def ask(question: str, llm) -> dict:
    retrieved = retrieve(question)
    prompt = build_prompt(question, retrieved)
    answer = generate_answer(prompt, llm)
    return {
        'question': question,
        'answer': answer,
        'sources': retrieved
    }


# ── EVALUATION ──────────────────────────────────────────────
def run_evaluation(llm):
    test_questions = [
        "Why are people unhappy with credit cards?",
        "What are the most common issues with personal loans?",
        "What problems do customers face with money transfers?",
        "Are there fraud complaints in savings accounts?",
        "What billing issues do credit card customers report?",
        "Why do customers complain about loan repayment?",
        "What are the most reported issues with checking accounts?",
    ]
    print("\n" + "="*60)
    print("RAG PIPELINE EVALUATION")
    print("="*60)
    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}/{len(test_questions)}] Q: {question}")
        result = ask(question, llm)
        print(f"A: {result['answer']}")
        print(f"Sources: {len(result['sources'])} chunks retrieved")
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    llm = load_llm()
    run_evaluation(llm)