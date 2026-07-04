import streamlit as st
from langchain_ollama import OllamaLLM


# ======================================================
# LOAD OLLAMA MODEL
# ======================================================

@st.cache_resource
def load_llm():
    """
    Loads the Ollama Llama 3.2 model.
    The model is cached so it loads only once.
    """

    llm = OllamaLLM(
        model="llama3.2",
        temperature=0
    )

    return llm