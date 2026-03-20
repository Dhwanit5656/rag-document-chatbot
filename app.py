# main.py
# Streamlit UI — uses pipeline.py as backend

import streamlit as st
import os
import tempfile
from pipeline import (
    load_document_chunk,
    vector_store,
    search_query,
    get_answer,
    detect_language
)

# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multilingual-MultiDoc RAG Chatbot",
    page_icon="🗣️",
    layout="wide"
)

# ── Session State ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = []

if "is_ready" not in st.session_state:
    st.session_state.is_ready = False





# ── Header ─────────────────────────────────────────────────────────
st.title("🗣️ Multilingual-MultiDoc RAG Chatbot")
st.caption("Chat with your documents in Hindi or English")


# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Documents")
    st.caption("Supports PDF, TXT, DOCX")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    col1, col2 = st.columns(2)

    if col1.button("🚀 Process", type="primary", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing documents..."):

                temp_paths = []
                name_map = {}  # maps temp filename → original filename

            for file in uploaded_files:
                suffix = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as tmp:
                    tmp.write(file.read())
                    temp_paths.append(tmp.name)
                    # Remember: temp path → original name
                    name_map[os.path.basename(tmp.name)] = file.name

            # Run pipeline
            chunks = load_document_chunk(temp_paths)

            # Fix source names — replace temp names with original names
            for chunk in chunks:
                temp_source = chunk.metadata.get("source", "")
                if temp_source in name_map:
                    chunk.metadata["source"] = name_map[temp_source]

            vector_store(chunks)

            # Save file names for sidebar display
            st.session_state.docs_loaded = [f.name for f in uploaded_files]
            st.session_state.is_ready = True
            st.session_state.messages = []

            # Delete temp files
            for path in temp_paths:
                os.unlink(path)

        st.success(f"✅ {len(uploaded_files)} file(s) ready!")
    else:
        st.warning("Please upload at least one file")

    if col2.button("🗑️ Reset", use_container_width=True):
        st.session_state.messages = []
        st.session_state.docs_loaded = []
        st.session_state.is_ready = False
        st.rerun()

    # Show loaded files in sidebar
    if st.session_state.docs_loaded:
        st.divider()
        st.subheader("📋 Loaded Files")
        for doc in st.session_state.docs_loaded:
            st.caption(f"✅ {doc}")

    if st.session_state.is_ready:
        st.divider()
        st.caption(f"💬 Messages: {len(st.session_state.messages)}")


# ── Chat History Display ───────────────────────────────────────────
for msg in st.session_state.messages:

    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])

    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

        

            if "sources" in msg:
                with st.expander("📄 Sources"):
                    for meta in msg["sources"]:
                        flag = "🇮🇳" if meta.get("language") == "hi" else "🇬🇧"
                        st.caption(
                            f"{flag} {meta.get('source', 'unknown')} "
                            f"| Page: {meta.get('page', 'N/A')}"
                        )


# ── Chat Input ─────────────────────────────────────────────────────
if question := st.chat_input(
    "Ask in Hindi or English...",
    disabled=not st.session_state.is_ready
):
    # Detect language
    language = detect_language(question)
    flag = "🇮🇳 Hindi" if language == "hi" else "🇬🇧 English"

    # Show user message
    display_question = f"{question} `{flag}`"
    with st.chat_message("user"):
        st.markdown(display_question)

    st.session_state.messages.append({
        "role": "user",
        "content": display_question
    })

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("🔄 Searching..."):
            contexts, metas = search_query(question)

        with st.spinner("🤔 Thinking..."):
            response = get_answer(question, contexts, language)
            st.markdown(response)

        

        

        with st.expander("📄 Sources"):
            seen = set()
            for meta in metas:
                source = meta.get("source", "unknown")
                if source not in seen:
                    lang_flag = "🇮🇳" if meta.get("language") == "hi" else "🇬🇧"
                    st.caption(
                        f"{lang_flag} {source} "
                        f"| Page: {meta.get('page', 'N/A')}"
                    )
                    seen.add(source)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": metas
    })


# ── Empty State ────────────────────────────────────────────────────
if not st.session_state.is_ready:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Step 1**\n\nUpload PDF, TXT or DOCX files in the sidebar")
    with c2:
        st.info("**Step 2**\n\nClick Process to embed your documents")
    with c3:
        st.info("**Step 3**\n\nAsk questions in Hindi or English!")
