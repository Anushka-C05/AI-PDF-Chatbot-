from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_pdf(uploaded_file):
    """
    Reads the uploaded PDF, extracts text,
    splits it into chunks and returns:

    reader
    extracted text
    chunks
    """

    # ----------------------------------------
    # Read PDF
    # ----------------------------------------

    reader = PdfReader(uploaded_file)

    text = ""

    # ----------------------------------------
    # Extract Text
    # ----------------------------------------

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    # ----------------------------------------
    # Check if PDF contains text
    # ----------------------------------------

    if text.strip() == "":

        raise ValueError(
            "This PDF contains no readable text."
        )

    # ----------------------------------------
    # Split into Chunks
    # ----------------------------------------

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=500,

        chunk_overlap=100,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]

    )

    documents = []

    for page_num, page in enumerate(reader.pages):

        page_text = page.extract_text()

        if page_text:

           page_chunks = splitter.split_text(page_text)
           for chunk in page_chunks:

            documents.append({
                "page": page_num + 1,
                "text": chunk
            })

    chunks = splitter.split_text(text)

    return reader, text, chunks
   
 
        