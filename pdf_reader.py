from pypdf import PdfReader

# Function to read PDF
def read_pdf(file_path):

    # Open the PDF
    reader = PdfReader(file_path)

    # Store all text here
    text = ""

    # Read each page
    for page in reader.pages:
        text += page.extract_text()

    return text


# Test the function
pdf_text = read_pdf("pdfs/sample.pdf")

print(pdf_text)