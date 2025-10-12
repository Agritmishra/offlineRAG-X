# core.py
# Advanced Offline RAG core with persistence, KG, and optional local LLM summarizer support.
import os
import re
import zipfile
import json
import tempfile
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import networkx as nx
import matplotlib.pyplot as plt

# Optional imports for local summarizer (lazy)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

EMBED_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2"
}

def split_text_to_chunks(text: str, chunk_words: int = 400, overlap: int = 50) -> List[str]:
    tokens = text.split()
    if len(tokens) <= chunk_words:
        return [text]
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = " ".join(tokens[i:i + chunk_words])
        chunks.append(chunk)
        i += chunk_words - overlap
    return chunks

def extract_text_from_pdf(path: str) -> str:
    text_parts = []
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    except Exception:
        return ""
    return "\\n".join(text_parts)

def extract_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    else:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

class AdvancedRAG:
    def __init__(self, embed_model_key: str = "all-MiniLM-L6-v2"):
        assert embed_model_key in EMBED_MODELS
        self.model_key = embed_model_key
        self.embed_model_name = EMBED_MODELS[embed_model_key]
        self.embed_model = SentenceTransformer(self.embed_model_name)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[str] = []
        self.metadatas: List[Tuple[str, str]] = []  # (source_filename, excerpt)
        self.embeddings: Optional[np.ndarray] = None
        # Summarizer optional (seq2seq)
        self.summarizer = None
        self.summarizer_model_id = None

    def switch_model(self, embed_model_key: str):
        if embed_model_key == self.model_key:
            return
        assert embed_model_key in EMBED_MODELS
        self.model_key = embed_model_key
        self.embed_model_name = EMBED_MODELS[embed_model_key]
        self.embed_model = SentenceTransformer(self.embed_model_name)
        if self.chunks:
            self._rebuild_index(self.chunks, self.metadatas)

    def _embed(self, texts: List[str]) -> np.ndarray:
        embs = self.embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(embs)
        return embs

    def _rebuild_index(self, chunks: List[str], metadatas: List[Tuple[str,str]]):
        embs = self._embed(chunks)
        d = embs.shape[1]
        idx = faiss.IndexFlatIP(d)
        idx.add(embs)
        self.index = idx
        self.embeddings = embs
        self.chunks = chunks
        self.metadatas = metadatas

    def ingest(self, filepaths: List[str], chunk_words: int = 400, overlap: int = 50):
        all_chunks = []
        metas = []
        for p in filepaths:
            text = extract_text_from_file(p)
            if not text or len(text.strip()) == 0:
                continue
            chunks = split_text_to_chunks(text, chunk_words=chunk_words, overlap=overlap)
            for c in chunks:
                all_chunks.append(c)
                metas.append((os.path.basename(p), c[:200]))
        if not all_chunks:
            raise ValueError("No text extracted from files.")
        self._rebuild_index(all_chunks, metas)

    def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        if not self.index or self.embeddings is None or len(self.chunks) == 0:
            return []
        q_emb = self._embed([query])
        D, I = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            results.append({
                "score": float(score),
                "chunk": self.chunks[idx],
                "source": self.metadatas[idx][0],
                "excerpt": self.metadatas[idx][1]
            })
        return results

    def summarize_text(self, text: str, summary_ratio: float = 0.25) -> str:
        sentences = re.split(r'(?<=[.!?]) +', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if not sentences:
            return text[:800]
        sent_embs = self._embed(sentences)
        doc_emb = np.mean(sent_embs, axis=0, keepdims=True)
        sims = cosine_similarity(sent_embs, doc_emb).flatten()
        top_n = max(1, int(len(sentences) * summary_ratio))
        top_idx = sorted(list(np.argsort(sims)[-top_n:]))
        selected = [sentences[i] for i in top_idx]
        return " ".join(selected)

    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        vec = TfidfVectorizer(stop_words='english', max_features=500)
        X = vec.fit_transform([text])
        arr = X.toarray()[0]
        if arr.sum() == 0:
            return []
        top_idx = arr.argsort()[-top_k:][::-1]
        feat = vec.get_feature_names_out()
        return [feat[i] for i in top_idx]

    def answer(self, question: str, k: int = 4, summary_ratio: float = 0.3) -> Dict[str, Any]:
        hits = self.retrieve(question, k)
        if not hits:
            return {"answer": "No indexed documents found. Upload files and index first.", "evidence": []}
        combined = " ".join([h["chunk"] for h in hits])
        answer = self.summarize_text(combined, summary_ratio)
        highlighted = []
        q_emb = self._embed([question])
        for h in hits:
            sentences = re.split(r'(?<=[.!?]) +', h["chunk"])
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
            if sentences:
                sent_embs = self._embed(sentences)
                sims = cosine_similarity(sent_embs, q_emb).flatten()
                best_i = int(np.argmax(sims))
                highlighted.append({"source": h["source"], "best_sentence": sentences[best_i], "score": float(np.max(sims))})
            else:
                highlighted.append({"source": h["source"], "best_sentence": h["chunk"][:200], "score": 0.0})
        return {"answer": answer, "evidence": hits, "highlights": highlighted}

    
    # Persistence: save/load index and metadata
    
    def save_index(self, path: str):
        os.makedirs(path, exist_ok=True)
        if self.index is None or self.embeddings is None:
            raise ValueError("No index to save.")
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadatas, f)

    def load_index(self, path: str):
        idx_file = os.path.join(path, "index.faiss")
        emb_file = os.path.join(path, "embeddings.npy")
        meta_file = os.path.join(path, "meta.json")
        if not (os.path.exists(idx_file) and os.path.exists(emb_file) and os.path.exists(meta_file)):
            raise ValueError("Index files missing.")
        self.index = faiss.read_index(idx_file)
        self.embeddings = np.load(emb_file)
        with open(meta_file, "r", encoding="utf-8") as f:
            self.metadatas = json.load(f)
        self.chunks = [m[1] for m in self.metadatas]

    def export_index_zip(self, zip_path: str):
        tmpdir = tempfile.mkdtemp()
        self.save_index(tmpdir)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(tmpdir):
                zf.write(os.path.join(tmpdir, fname), arcname=fname)

    def import_index_zip(self, zip_path: str):
        tmpdir = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)
        self.load_index(tmpdir)

    
    # Knowledge graph: doc-to-doc similarity graph
    
    def build_doc_graph(self, similarity_threshold: float = 0.7) -> nx.Graph:
        file_chunks = {}
        for (src, _), chunk in zip(self.metadatas, self.chunks):
            file_chunks.setdefault(src, []).append(chunk)
        file_embs = {}
        for fname, chunks in file_chunks.items():
            embs = self._embed(chunks)
            centroid = embs.mean(axis=0)
            file_embs[fname] = centroid
        fnames = list(file_embs.keys())
        G = nx.Graph()
        for f in fnames:
            G.add_node(f)
        for i in range(len(fnames)):
            for j in range(i+1, len(fnames)):
                sim = float(np.dot(file_embs[fnames[i]], file_embs[fnames[j]]))
                if sim >= similarity_threshold:
                    G.add_edge(fnames[i], fnames[j], weight=sim)
        return G

    def draw_graph(self, G: nx.Graph, figsize=(6,6)):
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G, seed=42)
        weights = [G[u][v]['weight'] for u,v in G.edges()]
        nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color=weights, width=2.0, edge_cmap=plt.cm.Blues, node_size=900)
        plt.tight_layout()
        fig = plt.gcf()
        return fig

    
    # Optional local summarizer (seq2seq) for abstractive rewriting
    
    def init_local_summarizer(self, model_id: Optional[str] = None, device: int = -1):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers not available in this environment.")
        model_id = model_id or os.getenv("LOCAL_SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        pipe = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)
        self.summarizer = pipe
        self.summarizer_model_id = model_id

    def local_summarize(self, text: str, max_length: int = 200, min_length: int = 60) -> str:
        if self.summarizer is None:
            raise RuntimeError("Local summarizer not initialized. Call init_local_summarizer() first.")
        # The summarization pipeline may have token limits; chunk if needed
        try:
            res = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            if isinstance(res, list) and len(res) > 0 and "summary_text" in res[0]:
                return res[0]["summary_text"]
            # fallback
            return str(res)
        except Exception as e:
            # fallback to extractive if summarizer fails
            return text[:max_length]
