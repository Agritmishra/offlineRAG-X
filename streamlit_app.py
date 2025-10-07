# streamlit_app.py
import streamlit as st
import os
import tempfile
import matplotlib.pyplot as plt
from core import AdvancedRAG, EMBED_MODELS
from typing import List
import base64

st.set_page_config(page_title="Advanced Offline RAG — Summarizer", layout="wide")
st.title("Advanced Offline RAG — Privacy-first (LLM Summarizer Included)")

# Session state
if "rag" not in st.session_state:
    st.session_state.rag = AdvancedRAG()
if "uploaded_paths" not in st.session_state:
    st.session_state.uploaded_paths = []

# Sidebar: ingestion + settings
with st.sidebar:
    st.header("Indexing & Settings")
    st.markdown("Upload TXT or PDF files. Indexing rebuilds the index from uploaded files.")
    uploaded = st.file_uploader("Upload files (pdf / txt)", accept_multiple_files=True, type=["pdf", "txt"])
    model_choice = st.selectbox("Embedding model (local)", list(EMBED_MODELS.keys()), index=0)
    chunk_words = st.slider("Chunk size (words)", 200, 800, 400, step=50)
    overlap = st.slider("Chunk overlap (words)", 0, 200, 50, step=10)

    st.markdown("### Advanced features (optional)")
    enable_persistence = st.checkbox("Enable index persistence (save/load)", value=False)
    enable_export = st.checkbox("Enable index export/import (.zip)", value=False)
    enable_kg = st.checkbox("Enable knowledge-graph view", value=False)
    enable_local_summarizer_ui = st.checkbox("Enable local LLM summarizer (optional)", value=False)
    local_summarizer_model = st.text_input("Local summarizer model id (optional)", value="sshleifer/distilbart-cnn-12-6" if enable_local_summarizer_ui else "")
    if st.button("Build Index"):
        if not uploaded:
            st.warning("Upload at least one file before building index.")
        else:
            tmp = tempfile.mkdtemp()
            paths = []
            for f in uploaded:
                path = os.path.join(tmp, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                paths.append(path)
            with st.spinner("Ingesting and building index..."):
                st.session_state.rag.switch_model(model_choice)
                try:
                    st.session_state.rag.ingest(paths, chunk_words=chunk_words, overlap=overlap)
                    st.session_state.uploaded_paths = paths
                except Exception as e:
                    st.error(f"Ingest failed: {e}")
                    st.stop()
            st.success("Index built ✔")
    if enable_persistence and st.button("Save Index to disk (saved in /tmp)"):
        try:
            path = "/tmp/rag_index"
            st.session_state.rag.save_index(path)
            st.success(f"Saved to {path}")
        except Exception as e:
            st.error(f"Save failed: {e}")
    if enable_export and st.session_state.rag.index is not None and st.button("Export Index (.zip)"):
        try:
            outzip = "/tmp/rag_index.zip"
            st.session_state.rag.export_index_zip(outzip)
            with open(outzip, "rb") as f:
                b = f.read()
            b64 = base64.b64encode(b).decode()
            href = f'<a href="data:application/zip;base64,{b64}" download="rag_index.zip">Download exported index</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Export failed: {e}")
    if enable_export and st.file_uploader("Or import index (.zip)", type=["zip"]):
        z = st.file_uploader("Upload index zip", type=["zip"])
        if z:
            tmpf = os.path.join(tempfile.mkdtemp(), z.name)
            with open(tmpf, "wb") as out:
                out.write(z.getbuffer())
            try:
                st.session_state.rag.import_index_zip(tmpf)
                st.success("Index imported from zip.")
            except Exception as e:
                st.error(f"Import failed: {e}")

    st.markdown("---")
    st.write("Index status:")
    if st.session_state.rag.chunks:
        st.write(f"Indexed chunks: {len(st.session_state.rag.chunks)}")
        files = [m[0] for m in st.session_state.rag.metadatas]
        counts = {}
        for fn in files:
            counts[fn] = counts.get(fn, 0) + 1
        top_files = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
        st.write("Top files:", ", ".join([f"{k} ({v})" for k,v in top_files]))
    else:
        st.write("No index.")

# Main UI
st.header("Modes")
mode = st.radio("Choose mode", ["Question Answering", "Summarize All", "Explore Index"])

if mode == "Summarize All":
    st.subheader("Document Summaries")
    if not st.session_state.rag.chunks:
        st.info("No index. Upload files and click 'Build Index' first.")
    else:
        detail_option = st.selectbox(
            "Summary Detail Level",
            ["Short (25%)", "Medium (40%)", "Detailed (60%)"],
            index=0
        )
        ratio_map = {
            "Short (25%)": 0.25,
            "Medium (40%)": 0.40,
            "Detailed (60%)": 0.60
        }
        summary_ratio = ratio_map[detail_option]
        use_llm = st.checkbox("Use LLM abstractive summarizer (optional)", value=False)
        llm_max = st.slider("LLM max summary tokens (approx words)", 60, 400, 200)
        for fname, chunk_list in ((lambda m,c: list({}.items())) if False else []):
            pass
        files_map = {}
        for (src, _), chunk in zip(st.session_state.rag.metadatas, st.session_state.rag.chunks):
            files_map.setdefault(src, []).append(chunk)
        for fname, chunk_list in files_map.items():
            combined = " ".join(chunk_list)
            st.markdown(f"### {fname}")
            extractive = st.session_state.rag.summarize_text(combined, summary_ratio=summary_ratio)
            st.write("**Extractive summary**")
            st.write(extractive)
            if use_llm:
                if not st.session_state.rag.summarizer:
                    st.info("Local summarizer not initialized. Initialize in sidebar (Enable local LLM summarizer).")
                else:
                    try:
                        abstr = st.session_state.rag.local_summarize(extractive, max_length=llm_max, min_length=max(40, int(llm_max*0.2)))
                        st.write("**LLM abstractive summary**")
                        st.write(abstr)
                    except Exception as e:
                        st.error(f"LLM summarization failed: {e}")
            keywords = st.session_state.rag.extract_keywords(combined, top_k=8)
            if keywords:
                st.write("**Keywords:**", ", ".join(keywords))
            st.markdown("---")

elif mode == "Question Answering":
    st.subheader("Ask a question")
    question = st.text_input("Enter question")
    k = st.slider("How many retrieved chunks (k)", 1, 8, 4)
    # Provide same clear detail options for QA answer summarization
    qa_detail = st.selectbox("Answer detail level", ["Short (25%)", "Medium (40%)", "Detailed (60%)"], index=0)
    ratio_map = {"Short (25%)":0.25, "Medium (40%)":0.40, "Detailed (60%)":0.60}
    summary_ratio = ratio_map[qa_detail]
    enable_llm_rewrite = st.checkbox("Use LLM to rewrite final answer (optional)", value=False)
    if enable_llm_rewrite and not st.session_state.rag.summarizer:
        st.info("If you enable LLM rewrite, initialize the local summarizer in the sidebar first.")
    if st.button("Get Answer"):
        if not question:
            st.warning("Type a question.")
        else:
            if not st.session_state.rag.chunks:
                st.info("No index found. Upload files and build index first.")
            else:
                with st.spinner("Retrieving and composing answer..."):
                    out = st.session_state.rag.answer(question, k=k, summary_ratio=summary_ratio)
                st.markdown("#### Answer (extractive)")
                st.write(out["answer"])
                st.markdown("#### Evidence (retrieved chunks)")
                scores = [h["score"] for h in out["evidence"]]
                fig, ax = plt.subplots()
                ax.barh(range(len(scores))[::-1], scores[::-1])
                ax.set_yticks(range(len(scores)))
                ax.set_yticklabels([f"{h['source']}" for h in out["evidence"]][::-1])
                ax.set_xlabel("Similarity score")
                st.pyplot(fig)
                for i, h in enumerate(out["evidence"]):
                    st.markdown(f"**{i+1}. Source:** {h['source']} — **score:** {h['score']:.4f}")
                    with st.expander("Show chunk / excerpt"):
                        st.write(h["chunk"][:2000])
                st.markdown("#### Highlighted sentences (most relevant to question)")
                for hl in out["highlights"]:
                    st.markdown(f"- **{hl['source']}** (score: {hl['score']:.3f}): {hl['best_sentence']}")
                if enable_llm_rewrite:
                    if not st.session_state.rag.summarizer:
                        st.error("Local summarizer not initialized.")
                    else:
                        prompt_text = out["answer"]
                        try:
                            abstr = st.session_state.rag.local_summarize(prompt_text, max_length=200, min_length=60)
                            st.markdown("#### LLM-rewritten answer (abstractive)")
                            st.write(abstr)
                        except Exception as e:
                            st.error(f"LLM rewrite failed: {e}")

else:  # Explore Index
    st.subheader("Index Explorer")
    if not st.session_state.rag.chunks:
        st.info("No index.")
    else:
        st.markdown("### Knowledge Graph (file-level similarity)")
        threshold = st.slider("Edge similarity threshold (0-1)", 0.5, 0.95, 0.72)
        try:
            G = st.session_state.rag.build_doc_graph(similarity_threshold=threshold)
            if len(G.nodes()) == 0:
                st.write("Graph empty (maybe few files).")
            else:
                fig = st.session_state.rag.draw_graph(G, figsize=(6,6))
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Graph build failed: {e}")
        st.markdown("---")
        st.markdown("### Index contents (first 20 chunks)")
        for i, (m, chunk) in enumerate(zip(st.session_state.rag.metadatas[:20], st.session_state.rag.chunks[:20])):
            st.markdown(f"**{i+1}. File:** {m[0]} — excerpt: {m[1]}")
            with st.expander("Show chunk"):
                st.write(chunk[:1500])

st.markdown("---")
st.caption("Advanced Offline RAG — LLM summarizer optional. Initialize summarizer only if you have enough RAM.")

# Initialize local summarizer button in sidebar area (outside context)
with st.sidebar:
    if st.checkbox("Initialize local summarizer now (download model)", value=False):
        model_id_env = os.getenv("LOCAL_SUMMARIZER_MODEL")
        model_id = model_id_env if model_id_env else (st.session_state.get('local_summarizer_model') or "sshleifer/distilbart-cnn-12-6")
        try:
            st.session_state.rag.init_local_summarizer(model_id=model_id)
            st.success(f"Local summarizer initialized: {st.session_state.rag.summarizer_model_id}")
        except Exception as e:
            st.error(f"Failed to init summarizer: {e}")
