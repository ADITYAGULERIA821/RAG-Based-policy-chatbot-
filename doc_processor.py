# app/doc_processor.py Loads a PDF, splits it into chunks (to later embed in vector DB).

# app/doc_processor.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(pdf_path):
    """
    Load PDF and split into chunks of text for embedding.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    return chunks

# Quick test snippet (optional)
if __name__ == "__main__":
    chunks = load_and_chunk_pdf("data/Synise Handbook.pdf")
    print(f"Number of chunks: {len(chunks)}")
    print(chunks[0].page_content[:300])  # print first 300 chars of first chunk


 