from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st


# ======================================================
# LOAD EMBEDDING MODEL (Cached)
# ======================================================

@st.cache_resource
def load_embedding_model():
    """
    Loads the HuggingFace embedding model.
    The model is cached so it loads only once.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embedding_model


# ======================================================
# CREATE VECTOR DATABASE
# ======================================================

def create_vector_db(chunks):
    """
    Creates a FAISS vector database from text chunks.
    """

    embedding_model = load_embedding_model()

    vector_db = FAISS.from_texts(
        texts=chunks,
        embedding=embedding_model
    )

    return vector_db


# ======================================================
# SIMILARITY SEARCH
# ======================================================

def search_vector_db(vector_db, question, k=3):
    """
    Retrieves the top-k most relevant chunks
    from the vector database.
    """

    docs = vector_db.similarity_search(
        question,
        k=k
    )

    return docs