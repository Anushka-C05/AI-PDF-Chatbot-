from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS database
vector_db = FAISS.load_local(
    "vector_db",
    embedding_model,
    allow_dangerous_deserialization=True
)

# Load Ollama model
llm = OllamaLLM(model="llama3.2")

print("Chatbot is ready!")

# Ask question
question = input("\nAsk a question about the PDF: ")

# Retrieve relevant chunks
docs = vector_db.similarity_search(question, k=3)

# Build context
context = ""

for doc in docs:
    context += doc.page_content + "\n\n"

# Prompt
prompt = f"""
You are a helpful assistant.

Answer ONLY using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

# Generate answer
answer = llm.invoke(prompt)

print("\nAnswer:\n")
print(answer)