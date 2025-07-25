from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder

def embed_chunks(chunks):
    model = get_embedder()
    return model.encode(chunks, convert_to_numpy=True)

def save_to_faiss(chunks, embeddings, index_path="data/index.faiss", metadata_path="data/chunks.txt"):
    os.makedirs("data", exist_ok=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.strip() + "\n---\n")

def load_chunks(path="data/chunks.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().split("\n---\n")

def load_index(index_path="data/index.faiss"):
    return faiss.read_index(index_path)

def retrieve_top_k(query, k=3):
    model = get_embedder()
    index = load_index()
    chunks = load_chunks()

    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k)

    return [chunks[i] for i in I[0]]
