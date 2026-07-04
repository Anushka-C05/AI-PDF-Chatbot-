from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_reader import read_pdf

# Read PDF
text = read_pdf("pdfs/sample.pdf")

# Create splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Split text
chunks = splitter.split_text(text)

# Print results
print("Total Chunks:", len(chunks))

print("\nFirst Chunk:\n")
print(chunks[0])