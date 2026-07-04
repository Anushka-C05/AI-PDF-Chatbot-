import time
import streamlit as st
from PIL import Image

from pdf_utils import process_pdf
from vector_store import create_vector_db
from llm import load_llm
from rag import answer_question

# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

# ======================================================
# LOGO & HEADER
# ======================================================

logo = Image.open("assets/logo.png")

st.image(logo, width=120)

st.markdown("""
# 🤖 AI PDF Chatbot

Ask questions about your PDF documents using
**Retrieval-Augmented Generation (RAG)** powered by
**LangChain + FAISS + Ollama (Llama 3.2)**.
""")

st.divider()

# ======================================================
# SESSION STATE
# ======================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "reader" not in st.session_state:
    st.session_state.reader = None

if "text" not in st.session_state:
    st.session_state.text = ""

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ======================================================
# SIDEBAR
# ======================================================

with st.sidebar:

    st.title("📄 AI PDF Chatbot")

    st.markdown("---")

    st.subheader("🚀 Technologies Used")

    st.markdown("""
- Python
- Streamlit
- LangChain
- HuggingFace Embeddings
- FAISS
- Ollama
- Llama 3.2
- Retrieval-Augmented Generation (RAG)
""")

    st.markdown("---")

    if st.session_state.pdf_name:

        st.success(
            f"**Current PDF:**\n\n{st.session_state.pdf_name}"
        )

    st.markdown("---")

    st.metric(
        "Questions Asked",
        len(st.session_state.messages) // 2
    )

    st.markdown("---")

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        st.rerun()

    if st.button("📂 Upload Another PDF"):

        st.session_state.vector_db = None
        st.session_state.chunks = None
        st.session_state.reader = None
        st.session_state.pdf_name = None
        st.session_state.messages = []

        st.rerun()

# ======================================================
# DISPLAY OLD CHAT
# ======================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# ======================================================
# PDF UPLOAD
# ======================================================

uploaded_file = st.file_uploader(

    "📄 Upload your PDF",

    type=["pdf"]

)

# ======================================================
# PROCESS PDF
# ======================================================

if uploaded_file:

    if st.session_state.vector_db is None:

        pdf_start = time.time()

        try:

            with st.spinner("📄 Processing PDF..."):

                reader, text, chunks = process_pdf(
                    uploaded_file
                )

                vector_db = create_vector_db(
                    chunks
                )

            pdf_end = time.time()

            # Save everything
            st.session_state.reader = reader
            st.session_state.text = text
            st.session_state.chunks = chunks
            st.session_state.vector_db = vector_db
            st.session_state.pdf_name = uploaded_file.name

            st.success(
                f"✅ PDF processed successfully in {round(pdf_end-pdf_start,2)} seconds."
            )

        except Exception as e:

            st.error(f"❌ {e}")

            st.stop()

    # ---------------------------------------
    # Load Stored Objects
    # ---------------------------------------

    reader = st.session_state.reader
    chunks = st.session_state.chunks
    vector_db = st.session_state.vector_db

    st.divider()

    # ---------------------------------------
    # PDF Information
    # ---------------------------------------

    st.subheader("📊 PDF Information")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Pages",
        len(reader.pages)
    )

    col2.metric(
        "Chunks",
        len(chunks)
    )

    col3.metric(
        "Vectors",
        vector_db.index.ntotal
    )

    # ---------------------------------------
    # Expandable PDF Details
    # ---------------------------------------

    with st.expander("📖 View Extracted Text"):

        st.text_area(
            "Extracted Text",
            st.session_state.text,
            height=300
        )

    with st.expander("📚 View First Text Chunk"):

        if len(chunks) > 0:

            st.write(chunks[0])

    st.divider()

# ======================================================
# ASK QUESTIONS
# ======================================================

question = st.chat_input(
    "💬 Ask a question about the uploaded PDF..."
)

if question:

    # ---------------------------------------
    # Display User Message
    # ---------------------------------------

    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # ---------------------------------------
    # Load LLM
    # ---------------------------------------

    llm = load_llm()

    # ---------------------------------------
    # Generate Answer
    # ---------------------------------------

    answer_start = time.time()

    with st.spinner("🤖 Generating answer..."):

        answer, docs = answer_question(
            question,
            vector_db,
            llm
        )

    answer_end = time.time()

    # ---------------------------------------
    # Display Assistant Response
    # ---------------------------------------

    with st.chat_message("assistant"):

        st.markdown(answer)

        st.caption(
            f"⚡ Response generated in {round(answer_end-answer_start,2)} seconds."
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # ---------------------------------------
    # Display Retrieved Chunks
    # ---------------------------------------

    with st.expander("📚 Source Chunks Used"):

        for i, doc in enumerate(docs):

            st.markdown(f"### Chunk {i+1}")

            st.write(doc.page_content)

            st.divider()

# ======================================================
# DOWNLOAD CHAT HISTORY
# ======================================================

if len(st.session_state.messages) > 0:

    chat_history = ""

    for msg in st.session_state.messages:

        role = msg["role"].capitalize()

        chat_history += f"{role}:\n"

        chat_history += msg["content"]

        chat_history += "\n\n"

    st.download_button(

        label="📥 Download Chat History",

        data=chat_history,

        file_name="chat_history.txt",

        mime="text/plain"

    )

# ======================================================
# FOOTER
# ======================================================

st.divider()

st.markdown(
    """
<div style='text-align:center;'>

Built with ❤️ using
<b>Python</b>,
<b>Streamlit</b>,
<b>LangChain</b>,
<b>FAISS</b>,
<b>HuggingFace Embeddings</b>,
<b>Ollama</b> &
<b>Llama 3.2</b>

</div>
""",
unsafe_allow_html=True
)

st.caption(
    "© 2026 AI PDF Chatbot | Developed by Anushka Chaudhari"
)