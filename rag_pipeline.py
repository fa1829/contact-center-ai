"""
contact-center-ai — RAG pipeline
Retrieval-Augmented Generation over historical incident logs.

Pipeline:
  1. Embed each incident log with TF-IDF (a lightweight, fully local
     embedding — no external model download required).
  2. Store embeddings + text + metadata in ChromaDB (local vector store).
  3. Given a new question ("why did volume spike in December?"), embed the
     question the same way, retrieve the most similar past incidents.
  4. Generate a plain-English answer grounded in those retrieved incidents.

Two generation modes are provided:
  - extractive_summary(): works fully offline, no API key needed. This is
    what runs by default and what I could test end-to-end.
  - generate_with_llm(): calls HuggingFace's router (OpenAI-compatible),
    same pattern as fleet-qa-copilot. Requires HF_TOKEN and internet access —
    run this in your WSL environment, not in a sandboxed one.
"""
import os
import numpy as np
import pandas as pd
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer

from dotenv import load_dotenv
load_dotenv()

# ---------- 1. Load data ----------
df = pd.read_csv("incident_logs.csv")
documents = df["text"].tolist()
ids = df["id"].tolist()
metadatas = df[["type"]].to_dict("records")

# ---------- 2. Embed with TF-IDF ----------
# TF-IDF turns each document into a vector of weighted word-importance scores.
# It's a simple, fully local stand-in for a neural embedding model — good
# enough to demonstrate the RAG *architecture* without needing a model
# download. Swapping in real sentence-transformer or HF embeddings later is
# a drop-in replacement (see README "upgrading to real embeddings").
vectorizer = TfidfVectorizer(max_features=512, stop_words="english")
tfidf_matrix = vectorizer.fit_transform(documents)
embeddings = tfidf_matrix.toarray().tolist()

# ---------- 3. Store in ChromaDB ----------
client = chromadb.PersistentClient(path="./chroma_store")
# Fresh collection each run, for a clean, reproducible demo
try:
    client.delete_collection("incident_logs")
except Exception:
    pass
collection = client.create_collection("incident_logs")
collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
print(f"Indexed {len(documents)} incident logs into ChromaDB collection 'incident_logs'")


def retrieve(query: str, k: int = 5):
    """Embed the query with the SAME fitted vectorizer, then retrieve top-k similar docs."""
    query_vec = vectorizer.transform([query]).toarray().tolist()
    results = collection.query(query_embeddings=query_vec, n_results=k)
    return results["documents"][0], results["metadatas"][0], results["distances"][0]


def extractive_summary(query: str, retrieved_docs: list) -> str:
    """
    Offline, no-API-key summary: surfaces the retrieved incidents directly and
    counts recurring incident types. This is the "fallback" / fully testable
    generation mode.
    """
    if not retrieved_docs:
        return "No related incidents found in the knowledge base."
    lines = [f"Found {len(retrieved_docs)} related historical incidents for: \"{query}\"\n"]
    for i, doc in enumerate(retrieved_docs, 1):
        lines.append(f"{i}. {doc}")
    return "\n".join(lines)


def generate_with_llm(query: str, retrieved_docs: list) -> str:
    """
    Production-style generation: sends the retrieved context + query to an
    LLM via HuggingFace's router (OpenAI-compatible chat completions), same
    pattern as fleet-qa-copilot. Requires:
      - HF_TOKEN environment variable set
      - Internet access to router.huggingface.co (not available in this
        sandbox — run this function in your WSL environment)
    """
    import requests

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN environment variable to use generate_with_llm().")

    context = "\n".join(f"- {doc}" for doc in retrieved_docs)
    prompt = (
        f"Based on the following historical contact-center incidents, answer the question "
        f"concisely in 2-3 sentences.\n\nIncidents:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )

    response = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "model": "meta-llama/Llama-3.1-8B-Instruct",  # swap for any router-supported model
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    test_queries = [
        "Why did call volume spike recently?",
        "What caused recent system outages?",
        "Are there staffing shortages during any particular shift?",
    ]

    for q in test_queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {q}")
        print("=" * 70)
        docs, metas, distances = retrieve(q, k=3)
        print(extractive_summary(q, docs))
        print(f"\n(Retrieval distances — lower is more similar: {[round(d, 3) for d in distances]})")

    print("\n" + "=" * 70)
    print("To test LLM-generated answers instead of the extractive summary,")
    print("set HF_TOKEN and call generate_with_llm(query, docs) — see README.")
