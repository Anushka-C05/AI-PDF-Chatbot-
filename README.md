# 🤖 AI PDF Chatbot

An AI-powered PDF Chatbot built using **Streamlit**, **LangChain**, **FAISS**, **HuggingFace Embeddings**, and **Ollama (Llama 3.2)**. This application allows users to upload PDF documents and ask context-aware questions using **Retrieval-Augmented Generation (RAG)**.

---

## 📌 Features

- 📄 Upload PDF documents
- 🔍 Extract text from PDFs
- ✂️ Split text into meaningful chunks
- 🧠 Generate embeddings using HuggingFace
- 📚 Store embeddings using FAISS Vector Database
- 🤖 Answer questions using Ollama (Llama 3.2)
- 💬 Chat interface with conversation history
- 📥 Download chat history
- 🗑️ Clear chat option
- 📊 Display PDF information (Pages, Chunks, Vectors)
- 📚 Display retrieved source chunks used to answer questions
- ⏳ Loading spinner during PDF processing and answer generation
- 🎨 Professional Streamlit user interface

---

## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain
- HuggingFace Embeddings
- FAISS
- Ollama
- Llama 3.2
- PyPDF
- Retrieval-Augmented Generation (RAG)

---

## 📂 Project Structure

```
AI-PDF-Chatbot/
│── app.py
│── pdf_utils.py
│── vector_store.py
│── rag.py
│── llm.py
│── requirements.txt
│── README.md
│── .gitignore
│
├── assets/
│   └── logo.png
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AI-PDF-Chatbot.git
```

### 2. Open the project

```bash
cd AI-PDF-Chatbot
```

### 3. Create a virtual environment

Windows

```bash
python -m venv venv
```

Activate

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🤖 Install Ollama

Download Ollama from:

https://ollama.com/download

After installing, pull the Llama 3.2 model:

```bash
ollama pull llama3.2
```

Run Ollama:

```bash
ollama serve
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 🚀 How It Works

1. Upload a PDF document.
2. Text is extracted from the PDF.
3. The extracted text is split into chunks.
4. HuggingFace generates embeddings for each chunk.
5. FAISS stores the embeddings for fast retrieval.
6. User enters a question.
7. Relevant chunks are retrieved using similarity search.
8. Ollama (Llama 3.2) generates an answer using the retrieved context.
9. The chatbot displays the answer along with the retrieved source chunks.

