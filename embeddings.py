from langchain_huggingface import HuggingFaceEmbeddings
from text_splitter import chunks

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create embeddings
embeddings = embedding_model.embed_documents(chunks)

# Print information
print("Total Chunks:", len(chunks))
print("Total Embeddings:", len(embeddings))
print("Length of One Embedding:", len(embeddings[0]))

# Print first embedding
print("\nFirst Embedding:")
print(embeddings[0])