


# RAG-Based-policy-chatbot-

Automated Policy QA Chatbot for Companies
🧠 Project Goal
Build an automated chatbot to answer company policy questions (HR, IT, Legal) using RAG + LLM so employees can quickly find accurate, contextual policy answers.
👤 User Journey
1. User types policy-related question
2. System fetches relevant chunk from documents (via vector search)
3. LLM generates contextual response
4. Option to view source or ask follow-ups
🧩 Implementation Phases
Phase 1: Upload & chunk policy docs
Phase 2: Embed & store in ChromaDB
Phase 3: RAG-based chatbot interface
Phase 4: Add source reference + PDF viewer
Phase 5: Admin panel for uploads


langchain>=0.1.17
langchain-community>=0.0.30
langchain-core>=0.1.45
langchain-huggingface>=0.0.1

chromadb>=0.4.24
sentence-transformers>=2.2.2
transformers>=4.41.1
torch>=2.2.2
google/flan-t5-base

pypdf>=3.17.1
streamlit>=1.34.0
faiss-cpu>=1.8.0  # optional
