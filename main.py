import streamlit as st
import os
import base64
from app.doc_processor import load_and_chunk_pdf
from app.embeddings import create_chroma_index
from app.rag_chain import build_rag_chain

def show_pdf(file_path):
    """Embed a PDF in the Streamlit app using base64 encoding."""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

# Initialize session state for chat history and vectorstore path
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = "db"

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

st.set_page_config(page_title="Policy QA Chatbot", page_icon="📄", layout="centered")
st.title("📄 Policy QA Chatbot")

# PDF uploader
uploaded_files = st.file_uploader(
    "Upload Policy PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="visible"
)

# Button to clear chat history
if st.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# Optional: Delete all PDFs
if st.button("🗑️ Delete All PDFs"):
    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            os.remove(os.path.join("data", file))
    st.success("All PDF files deleted from 'data/' folder.")

if uploaded_files:
    # Save all uploaded PDFs
    for pdf_file in uploaded_files:
        save_path = os.path.join("data", pdf_file.name)
        with open(save_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        st.success(f"📄 Saved {pdf_file.name}")
        show_pdf(save_path)  # Show PDF preview

    # Load and chunk all PDFs
    all_chunks = []
    for pdf_file in uploaded_files:
        file_path = os.path.join("data", pdf_file.name)
        chunks = load_and_chunk_pdf(file_path)
        all_chunks.extend(chunks)
    st.info(f"📚 Loaded {len(all_chunks)} chunks from uploaded PDFs.")

    # Create persistent Chroma index
    create_chroma_index(all_chunks, persist_dir=st.session_state.persist_dir)
    st.success("✅ Indexing complete and persisted.")

    # Build/reload QA chain
    st.session_state.qa_chain = build_rag_chain(persist_dir=st.session_state.persist_dir)
    st.success("🤖 QA Chain ready! Ask your questions below.")

else:
    if st.session_state.qa_chain is None:
        try:
            st.session_state.qa_chain = build_rag_chain(persist_dir=st.session_state.persist_dir)
            st.info("Loaded existing index and QA chain. You can start asking questions.")
        except Exception as e:
            st.warning("No index found. Upload PDFs to start.")

# Chat input
query = st.text_input("Enter your policy question:")

if query and st.session_state.qa_chain:
    with st.spinner("🤖 Generating answer..."):
        result = st.session_state.qa_chain.invoke(query)  # Updated to .invoke()
        answer = result["result"]
        sources = [doc.page_content[:300] + "..." for doc in result["source_documents"]]

        # Save in chat history
        st.session_state.chat_history.append({
            "user": query,
            "bot": answer,
            "sources": sources
        })

# Display chat messages in WhatsApp style
for chat in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(chat["bot"])
        if chat['sources']:
            with st.expander("📄 Sources"):
                for src in chat['sources']:
                    st.markdown(f"- {src}")












