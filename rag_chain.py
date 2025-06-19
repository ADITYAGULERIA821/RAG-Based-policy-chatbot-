from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load a better instruction-following LLM
def load_flan_llm():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        temperature=0.7
    )

    return HuggingFacePipeline(pipeline=pipe)


# Create and return the RAG QA chain
def build_rag_chain(persist_dir="db"):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Improved prompt
    prompt = PromptTemplate.from_template(
        """You are a concise and accurate assistant answering policy-related queries. Answer only using the information in the context.

Context:
{context}

Question:
{question}

Answer:"""
    )

    llm = load_flan_llm()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )


# Optional CLI for testing
if __name__ == "__main__":
    rag = build_rag_chain()
    print("Enter your policy question (or type 'exit'):")

    while True:
        query = input("> ")
        if query.lower() == "exit":
            break

        result = rag(query)
        print("\n🧠 Answer:")
        print(result['result'])

        print("\n📄 Sources:")
        for doc in result['source_documents']:
            print(doc.page_content[:300] + "...")
        print("-" * 50)










