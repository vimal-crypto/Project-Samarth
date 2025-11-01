# build_index.py
# Rebuild FAISS index from corpus.jsonl (optional helper)

import os
import json
import numpy as np
import pandas as pd

try:
    import faiss
except Exception:
    faiss = None

from sentence_transformers import SentenceTransformer

CORPUS_PATH = "corpus.jsonl"
INDEX_PATH  = "samarth_index.faiss"
EMB_PATH    = "samarth_embeddings.npy"
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"

def load_corpus(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append({"text": obj.get("text","")})
    return pd.DataFrame(rows)

def main():
    if faiss is None:
        raise RuntimeError("FAISS not installed. pip install faiss-cpu")

    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError("Missing corpus.jsonl. Run create_corpus.py first.")

    df = load_corpus(CORPUS_PATH)
    texts = df["text"].tolist()

    model = SentenceTransformer(MODEL_NAME)
    embs = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs.astype(np.float32))

    faiss.write_index(index, INDEX_PATH)
    np.save(EMB_PATH, embs)
    print("Index built: %s, embeddings saved: %s" % (INDEX_PATH, EMB_PATH))

if __name__ == "__main__":
    main()
