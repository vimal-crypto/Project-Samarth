# chatbot1.py
# Project Samarth — FAISS RAG Chatbot (1997–2017, APY + IMD)
# - Loads corpus.jsonl (facts generated from APY.csv + Sub_Division_IMD_2017.csv)
# - Builds/loads FAISS index with MiniLM embeddings
# - Answers natural-language questions with citations and retrieved facts
# - Streamlit UI suitable for Streamlit Community Cloud

import os
import json
import time
import math
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import faiss  # faiss-cpu
except Exception as e:
    faiss = None

from sentence_transformers import SentenceTransformer

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Project Samarth — Agri × Climate Chatbot", layout="wide")

CORPUS_PATH = "corpus.jsonl"          # produced by create_corpus.py (1997–2017 facts)
INDEX_PATH = "samarth_index.faiss"    # FAISS flat L2 index
EMB_PATH = "samarth_embeddings.npy"   # cached dense embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # fast & solid
TOP_K_DEFAULT = 8

# -----------------------------
# Utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def _load_corpus(corpus_path: str) -> pd.DataFrame:
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(
            f"Missing {corpus_path}. Run create_corpus.py first to generate it."
        )
    rows = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            meta = obj.get("meta", {})
            rows.append({
                "text": text,
                "state": (meta.get("state") or "").upper(),
                "crop":  (meta.get("crop") or "").upper(),
                "year":  meta.get("year"),
            })
    df = pd.DataFrame(rows)
    # guard
    if "text" not in df.columns or len(df) == 0:
        raise ValueError("corpus.jsonl is empty or malformed.")
    return df


@st.cache_resource(show_spinner=False)
def _load_model(model_name: str):
    return SentenceTransformer(model_name)


def _ensure_faiss():
    if faiss is None:
        st.error("FAISS not found. Ensure `faiss-cpu` is in requirements.txt.")
        st.stop()


def _normalize_query(q: str) -> str:
    return (q or "").strip()


def _summarize_years(years: List[int]) -> str:
    y = sorted(set(int(y) for y in years if str(y).isdigit()))
    if not y:
        return "N/A"
    return f"{y[0]}–{y[-1]}" if len(y) > 1 else str(y[0])


# -----------------------------
# Embedding + Index builders
# -----------------------------
def _embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 256) -> np.ndarray:
    embs = []
    n = len(texts)
    for i in range(0, n, batch_size):
        embs.append(model.encode(texts[i:i+batch_size], show_progress_bar=False, normalize_embeddings=True))
    return np.vstack(embs)


def _build_or_load_index(df: pd.DataFrame, model: SentenceTransformer) -> Tuple[Any, np.ndarray]:
    """
    Returns (faiss_index, embeddings)
    - If index & embeddings exist on disk, load them.
    - Otherwise, build from df['text'], save to disk, and return.
    """
    _ensure_faiss()

    if os.path.exists(INDEX_PATH) and os.path.exists(EMB_PATH):
        try:
            index = faiss.read_index(INDEX_PATH)
            embs = np.load(EMB_PATH)
            if index.ntotal != embs.shape[0]:
                raise ValueError("Index size and embeddings count mismatch; rebuilding.")
            return index, embs
        except Exception:
            st.warning("Existing index invalid — rebuilding now…")

    # Build new
    texts = df["text"].tolist()
    with st.spinner("Encoding corpus embeddings (MiniLM)…"):
        embs = _embed_texts(model, texts)  # (N, 384) normalized

    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine-sim because we normalized vectors
    index.add(embs.astype(np.float32))

    # Persist
    faiss.write_index(index, INDEX_PATH)
    np.save(EMB_PATH, embs)
    return index, embs


# -----------------------------
# Retrieval + Answering
# -----------------------------
def _search(index, model, query: str, df: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
    """
    Return a dataframe of top-k hits with columns: score, text, state, crop, year
    """
    q = _normalize_query(query)
    if not q:
        return pd.DataFrame(columns=["score", "text", "state", "crop", "year"])

    q_emb = model.encode([q], normalize_embeddings=True)
    D, I = index.search(q_emb.astype(np.float32), top_k)
    scores = D[0].tolist()
    idxs = I[0].tolist()

    # slice safely (invalid ids can be -1 if index is empty)
    hits = []
    for score, idx in zip(scores, idxs):
        if idx < 0 or idx >= len(df):
            continue
        row = df.iloc[idx]
        hits.append({
            "score": float(score),
            "text": row["text"],
            "state": row.get("state"),
            "crop":  row.get("crop"),
            "year":  row.get("year"),
        })
    return pd.DataFrame(hits)


def _compose_answer(hits: pd.DataFrame, user_q: str) -> str:
    """
    Option B: A natural-language paragraph that stitches together the top facts
    + a compact quantitative summary + source line.
    """
    if hits.empty:
        return "I couldn’t find relevant facts for that query within 1997–2017. Try rephrasing or include the state/crop/year."

    # Basic signal extraction
    states = [s for s in hits["state"].dropna().astype(str).tolist() if s]
    crops = [c for c in hits["crop"].dropna().astype(str).tolist() if c]
    years = [y for y in hits["year"].dropna().tolist() if str(y).isdigit()]

    state_phrase = ", ".join(sorted(set(states))) if states else "the requested region(s)"
    crop_phrase = ", ".join(sorted(set(crops))) if crops else "the relevant crop(s)"
    year_phrase = _summarize_years(years)

    # Make a readable paragraph
    # Keep it grounded — we only narrate what the retrieved facts say.
    bullets = []
    for _, r in hits.head(5).iterrows():
        # Keep each retrieved sentence as-is (already data-backed from corpus)
        bullets.append(f"- {r['text']}")

    para = (
        f"**Answer (1997–2017 window):**\n\n"
        f"From the most relevant facts, focusing on {state_phrase} and {crop_phrase} "
        f"across **{year_phrase}**, here’s what stands out:\n\n"
        + "\n".join(bullets) +
        "\n\n**Interpretation:** The retrieved facts combine both **production/area/yield** (from APY) "
        "and **annual rainfall (mm)** (from IMD). Patterns in your question (e.g., trend, comparison) can be judged "
        "by reading the production and rainfall values year by year in these facts.\n\n"
        "**Sources:** APY (District-wise crop production, 1997–2017) and IMD (Sub-division rainfall aggregated to state-year, up to 2017)."
    )
    return para


# -----------------------------
# UI
# -----------------------------
st.title("Project Samarth — Agri × Climate Chatbot (1997–2017)")
st.caption("Retrieval-Augmented QA over APY (production) and IMD (rainfall). Index = FAISS. Embeddings = MiniLM-L6-v2.")

with st.sidebar:
    st.header("Index & Filters")
    st.write("This chatbot answers using **1997–2017** facts only.")
    top_k = st.slider("Top-K retrieved facts", min_value=4, max_value=20, value=TOP_K_DEFAULT, step=1)
    st.markdown("---")
    st.write("**Index files**")
    st.code(f"{INDEX_PATH}\n{EMB_PATH}\n{CORPUS_PATH}")
    st.markdown("---")
    st.write("If you updated `corpus.jsonl`, press the button to rebuild.")
    if st.button("Rebuild Index"):
        if os.path.exists(INDEX_PATH): os.remove(INDEX_PATH)
        if os.path.exists(EMB_PATH): os.remove(EMB_PATH)
        st.success("Cleared old index. It will rebuild on next query.")

# Load resources
try:
    corpus_df = _load_corpus(CORPUS_PATH)
except Exception as e:
    st.error(f"Error loading corpus: {e}")
    st.stop()

model = _load_model(MODEL_NAME)
index, _ = _build_or_load_index(corpus_df, model)

# Corpus overview
with st.expander("Dataset overview", expanded=False):
    n = len(corpus_df)
    uniq_states = sorted(corpus_df["state"].dropna().unique().tolist())
    uniq_crops = sorted(corpus_df["crop"].dropna().unique().tolist())
    years = [y for y in corpus_df["year"].dropna().tolist() if str(y).isdigit()]
    st.write(f"**Corpus size:** {n:,} facts")
    st.write(f"**States:** {len(uniq_states)}")
    st.write(f"**Crops:** {len(uniq_crops)}")
    if years:
        st.write(f"**Year range:** {min(years)}–{max(years)}")
    st.dataframe(
        pd.DataFrame({
            "states_sample": uniq_states[:15],
            "crops_sample": uniq_crops[:15],
        })
    )

# Chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("table") is not None:
            st.dataframe(m["table"])
        if m.get("download") is not None:
            st.download_button(
                "Download retrieved facts (.csv)",
                m["download"].to_csv(index=False).encode("utf-8"),
                file_name="retrieved_facts.csv",
                mime="text/csv"
            )

user_q = st.chat_input("Ask anything about 1997–2017 crops & rainfall (e.g., 'Rice vs rainfall in Karnataka 2008–2015')")
if user_q:
    # User message
    st.session_state.chat.append({"role": "user", "content": user_q})

    # Retrieve
    with st.spinner("Searching knowledge base…"):
        hits_df = _search(index, model, user_q, corpus_df, top_k=top_k)

    # Compose
    answer = _compose_answer(hits_df, user_q)

    # Assistant message with table + download
    st.session_state.chat.append({
        "role": "assistant",
        "content": answer,
        "table": hits_df[["score", "year", "state", "crop", "text"]],
        "download": hits_df[["score", "year", "state", "crop", "text"]],
    })
    st.rerun()

st.markdown("---")
st.subheader("Citations")
st.write("""
- **APY (MoA&FW/DES)**: District-wise, season-wise crop production statistics (1997–2017 slice used here).
- **IMD (MoES)**: Sub-division monthly rainfall (aggregated to annual, mapped to states; available to 2017).
- **Note on method**: State rainfall is approximated as the mean of associated IMD sub-divisions in this prototype.
""")
