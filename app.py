import os, pickle, numpy as np, pandas as pd, faiss, gradio as gr
from transformers import pipeline as hf_pipeline

BASE_DIR = r'C:\Users\HP\Desktop\my files\rag-complaint-chatbot'
index = faiss.read_index(os.path.join(BASE_DIR, "vector_store", "complaints.index"))
chunks_df = pd.read_csv(os.path.join(BASE_DIR, "vector_store", "chunks_metadata.csv"))
with open(os.path.join(BASE_DIR, "vector_store", "vectorizer.pkl"), 'rb') as f:
    vectorizer, svd = pickle.load(f)
print("Loading LLM...")
llm = hf_pipeline("text-generation", model="gpt2", max_new_tokens=150)
print("Ready!")

def retrieve(query, k=5):
    q = svd.transform(vectorizer.transform([query])).astype(np.float32)
    _, indices = index.search(q, k)
    return [{'text': str(chunks_df.iloc[i]['chunk_text']), 'product_category': str(chunks_df.iloc[i]['product_category']), 'issue': str(chunks_df.iloc[i]['issue'])} for i in indices[0] if i != -1 and i < len(chunks_df)]

def answer_question(question):
    if not question.strip(): return "Please enter a question.", ""
    chunks = retrieve(question)
    context = "\n\n".join([f"[{c['product_category']}]: {c['text']}" for c in chunks])
    prompt = f"You are a financial analyst. Based on these complaints:\n{context}\nAnswer: {question}\nAnswer:"
    result = llm(prompt, max_new_tokens=100, do_sample=False, truncation=True, pad_token_id=50256)
    answer = result[0]['generated_text'].split("Answer:")[-1].strip()
    sources = "\n\n".join([f"**Source {i+1}** | `{c['product_category']}`\n> {c['text'][:200]}..." for i, c in enumerate(chunks)])
    return answer, sources

with gr.Blocks(theme=gr.themes.Soft(), title="CrediTrust Analyzer") as app:
    gr.Markdown("# 🏦 CrediTrust Financial — Complaint Analyzer\n### RAG-powered insights from real customer complaints\n---")
    question_input = gr.Textbox(label="💬 Ask a Question", placeholder="e.g. Why are people unhappy with credit cards?", lines=2)
    with gr.Row():
        submit_btn = gr.Button("🔍 Ask", variant="primary")
        clear_btn = gr.Button("🗑️ Clear")
    with gr.Row():
        answer_output = gr.Textbox(label="🤖 AI Answer", lines=6, interactive=False)
        sources_output = gr.Markdown(value="*Sources appear here after asking.*")
    gr.Examples(examples=[["Why are people unhappy with credit cards?"],["What problems do customers face with money transfers?"],["Are there fraud complaints in savings accounts?"],["What billing issues do credit card customers report?"]], inputs=question_input)
    submit_btn.click(fn=answer_question, inputs=question_input, outputs=[answer_output, sources_output])
    question_input.submit(fn=answer_question, inputs=question_input, outputs=[answer_output, sources_output])
    clear_btn.click(fn=lambda: ("", "*Sources appear here after asking.*"), outputs=[question_input, sources_output])

app.launch(inbrowser=True)
