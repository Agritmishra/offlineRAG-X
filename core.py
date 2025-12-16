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

# Optional for local summarizer
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

EMBED_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2", "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2"
}

def split_text_to_chunks(text: str, chunk_words: int = 400, overlap: int = 50) -> List[str]:
    tokens = text.split()
    if len(tokens) <= chunk_words:
        return [text]
    chunks = []
    i = 0
    step = chunk_words - overlap
    if step <= 0:
        raise ValueError("chunk overlap must be smaller than chunk size")

    while i < len(tokens):
        chunk = " ".join(tokens[i:i + chunk_words])
        chunks.append(chunk)
        i += step

    return chunks

def extract_text_from_pdf(path: str) -> str:
    
    # extract text from pdf file. if extraction fails, prints an error message.

    text_parts = []
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    except Exception as e:
        # Print to stderr (helpful when debugging in Streamlit / logs)
        print(f"extract_text_from_pdf failed for {path}: {e}")
        return ""
    return "\n".join(text_parts)


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
        self.metadatas: List[Tuple[str, str]] = []  
        self.embeddings: Optional[np.ndarray] = None
        # summarizer optional 
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
        
        #Produce embeddings using the SentenceTransformer
        
        embs = self.embed_model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False)
        embs = np.asarray(embs)
        # ensures float32 dtype for FAISS compatibility
        if embs.dtype != np.float32:
            embs = embs.astype(np.float32)
        # ensures contiguous memory layout
        if not embs.flags["C_CONTIGUOUS"]:
            embs = np.ascontiguousarray(embs, dtype=np.float32)
        # normalize for cosine similarity when using inner product index
        try:
            faiss.normalize_L2(embs)
        except Exception as e:
            raise RuntimeError(
                f"faiss.normalize_L2 failed: {e}. Embedding dtype: {embs.dtype}, shape: {embs.shape}")
        return embs


    def _rebuild_index(self, chunks: List[str], metadatas: List[Tuple[str, str]]):
        
        # rebuild FAISS index from chunks using batching.

        if not chunks:
            raise ValueError("No chunks provided to _rebuild_index.")
        batch_size = 256
        idx = None
        embeddings_list = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            embs = self._embed(batch)
            if embs is None or len(embs) == 0:
                continue
            if idx is None:
                d = embs.shape[1]
                idx = faiss.IndexFlatIP(d)
            idx.add(embs)
            embeddings_list.append(embs)
        if idx is None:
            raise ValueError("Failed to build FAISS index: no embeddings were generated.")
        if len(chunks) != len(metadatas):
            raise ValueError("Internal error: chunk–metadata mismatch")

        self.index = idx
        self.embeddings = np.vstack(embeddings_list).astype(np.float32)
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
        # no index -> no results
        if self.index is None or self.embeddings is None or not self.chunks:
            return []
            
        n = len(self.chunks)
        k = max(1, min(k, n))

        q_emb = self._embed([query])
        # ensures q_emb is a 2D, float32, contiguous array suitable for faiss.index.search
        q_emb = np.asarray(q_emb, dtype=np.float32)
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        if not q_emb.flags["C_CONTIGUOUS"]:
            q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)

        D, I = self.index.search(q_emb, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(D[0], I[0]):
            # faiss can return -1 when there are not enough items
            if idx < 0 or idx >= n:
                continue

            s = float(score)
            # cosine on normalized vectors must be in [-1, 1]
            if not np.isfinite(s) or s < -1.0 or s > 1.0:
                s = max(min(s, 1.0), -1.0)

            results.append({
                "score": s,
                "chunk": self.chunks[idx],
                "source": self.metadatas[idx][0],
                "excerpt": self.metadatas[idx][1]
            })

        return results



    def summarize_text(self, text: str, summary_ratio: float = 0.25) -> str:
        
        # sentence splitting. split on punctuation + whitespace or newlines
        raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 10]

        # if we can't split properly, just fallback
        if not sentences:
            return text[:800]

        # Very short text
        if len(sentences) <= 3:
            return " ".join(sentences)[:800]

        # Embed all sentences
        try:
            sent_embs = self._embed(sentences)  # normalized vectors
        except Exception:
            # If embedding fails for some reason, fallback
            return " ".join(sentences)[:800]
            
        sim_matrix = cosine_similarity(sent_embs, sent_embs)

        # build a graph.... nodes = sentences, edges weighted by similarity
        G = nx.Graph()
        n = len(sentences)
        for i in range(n):
            G.add_node(i)
        # Add edges for similar sentences 
        for i in range(n):
            for j in range(i + 1, n):
                weight = float(sim_matrix[i, j])
                if weight > 0.1: 
                    G.add_edge(i, j, weight=weight)

        # If graph is empty, fallback to centroid method
        if G.number_of_edges() == 0:
            doc_emb = np.mean(sent_embs, axis=0, keepdims=True)
            sims = cosine_similarity(sent_embs, doc_emb).flatten()
            top_n = max(1, int(len(sentences) * summary_ratio))
            top_n = min(top_n, max(1, len(sentences) - 1))
            top_idx = sorted(list(np.argsort(sims)[-top_n:]))
            return " ".join(sentences[i] for i in top_idx)

        # Pagerank over sentence graph
        try:
            scores = nx.pagerank(G, weight="weight")
        except Exception:
            # In case Pagerank explodes for some weird reason, fallback
            doc_emb = np.mean(sent_embs, axis=0, keepdims=True)
            sims = cosine_similarity(sent_embs, doc_emb).flatten()
            scores = {i: float(sims[i]) for i in range(len(sentences))}

        # how many sentences to keep
        top_n = max(1, int(len(sentences) * summary_ratio))
        # cap to ~60%.....
        top_n = min(top_n, max(1, int(len(sentences) * 0.6)))

        # Rank by score, then sort by original position to keep narrative flow
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        chosen_idx = sorted([i for i, _ in ranked])

        summary_sentences = [sentences[i] for i in chosen_idx]
        summary = " ".join(summary_sentences)
        # Safety cap to avoid insane length
        return summary[:2000]


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
            return {
                "answer": "No indexed documents found. Upload files and index first.",
                "evidence": [],
                "highlights": []
            }

        q_lower = question.lower().strip()
        is_definition_q = (
            q_lower.startswith("what is ")
            or q_lower.startswith("who is ")
            or q_lower.startswith("define ")
            or q_lower.startswith("explain ")
        )

        # Embed question once
        q_emb = self._embed([question])

        # tokens for simple lexical overlap
        q_tokens = [w for w in re.findall(r"\w+", q_lower) if len(w) > 2]
        q_token_set = set(q_tokens)

        highlighted = []
        candidate_sentences: List[str] = []

        # Track the single best global sentence for definition style Qs
        best_global_sent: Optional[str] = None
        best_global_score: float = -1.0

        for h in hits:
            sentences = re.split(r'(?<=[.!?])\s+|\n+', h["chunk"])
            sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

            if not sentences:
                highlighted.append({
                    "source": h["source"],
                    "best_sentence": h["chunk"][:200],
                    "score": 0.0
                })
                continue

            sent_embs = self._embed(sentences)
            sims = cosine_similarity(sent_embs, q_emb).flatten()

            n = len(sentences)
            scores = []

            for i, sent in enumerate(sentences):
                base_sim = float(sims[i])

                # Positional bonus....earlier sentences get a bit more weight
                if n > 1:
                    pos_bonus = (1.0 - i / (n - 1)) * 0.15
                else:
                    pos_bonus = 0.0

                # Lexical overlap bonus
                s_tokens = set(re.findall(r"\w+", sent.lower()))
                overlap = len(s_tokens & q_token_set)
                overlap_bonus = 0.04 * overlap

                # definitional bonus
                def_bonus = 0.0
                if is_definition_q and re.search(r"\bis\b", sent, re.IGNORECASE):
                    def_bonus = 0.18

                total_score = base_sim + pos_bonus + overlap_bonus + def_bonus
                scores.append(total_score)

                # track best sentence for definition questions
                if is_definition_q and total_score > best_global_score:
                    best_global_score = total_score
                    best_global_sent = sent

            scores = np.array(scores)

            # Best sentence for highlight in this chunk
            best_i = int(np.argmax(scores))
            highlighted.append({
                "source": h["source"],
                "best_sentence": sentences[best_i],
                "score": float(scores[best_i])
            })

            # for non-definition questions, we still build good sentences
            top_m = min(2, len(sentences))
            top_idx = scores.argsort()[-top_m:]

            for idx in top_idx:
                sent = sentences[idx]
                sent_tokens = set(re.findall(r"\w+", sent.lower()))
                if len(sent_tokens & q_token_set) > 0:
                    candidate_sentences.append(sent)
        #duplicate....
        candidate_sentences = list(dict.fromkeys(candidate_sentences))

        # Build final answer text
        if is_definition_q and best_global_sent:
            answer = best_global_sent

        else:
            if candidate_sentences:
                candidate_text = " ".join(candidate_sentences)
            else:
                candidate_text = " ".join([h["chunk"] for h in hits])
            answer = candidate_text


        return {
            "answer": answer,
            "evidence": hits,
            "highlights": highlighted
        }



  
    # save/load index and metadata

    def save_index(self, path: str):
        """
        Save faiss index, embeddings, metadata, and chunks to path
        """
        os.makedirs(path, exist_ok=True)
        if self.index is None or self.embeddings is None:
            raise ValueError("No index to save.")

        idx_path = os.path.join(path, "index.faiss")
        emb_path = os.path.join(path, "embeddings.npy")
        meta_path = os.path.join(path, "meta.json")
        chunks_path = os.path.join(path, "chunks.json")

        try:
            faiss.write_index(self.index, idx_path)
        except Exception as e:
            raise IOError(f"Failed to write faiss index to {idx_path}: {e}")

        try:
            np.save(emb_path, self.embeddings.astype(np.float32))
        except Exception as e:
            raise IOError(f"Failed to save embeddings to {emb_path}: {e}")

        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.metadatas, f)
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f)
        except Exception as e:
            raise IOError(f"Failed to write meta/chunks JSON to {path}: {e}")


    def load_index(self, path: str):
        idx_file = os.path.join(path, "index.faiss")
        emb_file = os.path.join(path, "embeddings.npy")
        meta_file = os.path.join(path, "meta.json")
        chunks_file = os.path.join(path, "chunks.json")

        if not (os.path.exists(idx_file) and os.path.exists(emb_file) and os.path.exists(meta_file)):
            raise ValueError("Index files missing.")

        # Load faiss index + embeddings
        self.index = faiss.read_index(idx_file)
        self.embeddings = np.load(emb_file)

        # load metadata
        with open(meta_file, "r", encoding="utf-8") as f:
            self.metadatas = json.load(f)

        # prefer full chunks if present, else fall back
        if os.path.exists(chunks_file):
            with open(chunks_file, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
        else:
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


    # Knowledge graph.....doc-to-doc similarity graph
    
    def build_doc_graph(self, similarity_threshold: float = 0.7) -> nx.Graph:
        """
        Build a file-level similarity graph. each file centroid is normalized.... edge weight = cosine similarity
        """
        file_chunks = {}
        n_meta = len(self.metadatas)
        n_chunks = len(self.chunks)
        n = min(n_meta, n_chunks)
        for i in range(n):
            src, _ = self.metadatas[i]
            chunk = self.chunks[i]
            file_chunks.setdefault(src, []).append(chunk)
        if n_meta != n_chunks:
            print(f"Warning: metadatas length ({n_meta}) != chunks length ({n_chunks})")

        # Compute centroids using precomputed embeddings
        file_embs = {}
        if self.embeddings is not None and len(self.embeddings) == len(self.metadatas):
            emb_groups = {}
            for i in range(len(self.metadatas)):
                fname, _ = self.metadatas[i]
                emb = self.embeddings[i]
                emb_groups.setdefault(fname, []).append(emb)
            for fname, emb_list in emb_groups.items():
                embs_arr = np.vstack(emb_list)
                centroid = embs_arr.mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                file_embs[fname] = centroid
        else:
            # Fallback..... embed chunks per file
            for fname, chunks in file_chunks.items():
                if not chunks:
                    continue
                embs = self._embed(chunks)
                if embs is None or len(embs) == 0:
                    continue
                centroid = embs.mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                file_embs[fname] = centroid

        # build graph using cosine similarity.....dot product of normalized centroids
        fnames = list(file_embs.keys())
        G = nx.Graph()
        for f in fnames:
            G.add_node(f)
        for i in range(len(fnames)):
            for j in range(i + 1, len(fnames)):
                a = file_embs[fnames[i]]
                b = file_embs[fnames[j]]
                sim = float(np.dot(a, b))
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

    
    # Optional local summarizer for abstractive rewriting
    
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
        # summarization may have token limits, chunk if needed
        try:
            res = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            if isinstance(res, list) and len(res) > 0 and "summary_text" in res[0]:
                return res[0]["summary_text"]
            # fallback
            return str(res)
        except Exception as e:
            # fallback to extractive if summarizer fails
            return text[:max_length]
