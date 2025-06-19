# app/embeddings.py

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_chroma_index(chunks, persist_dir="db"):
    """
    Create a persistent Chroma vector store index from chunks.
    """
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb

# Quick test snippet (optional)
if __name__ == "__main__":
    from app.doc_processor import load_and_chunk_pdf
    chunks = load_and_chunk_pdf("data/Synise Handbook.pdf")
    db = create_chroma_index(chunks)
    print("Index created with", db._collection.count())

