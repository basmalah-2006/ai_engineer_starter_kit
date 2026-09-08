from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

SCRIPT_DIR = Path(__file__).parent.resolve()
DB_DIR = SCRIPT_DIR / "chroma_db_multilingual"

embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small", 
    model_kwargs={'device': 'cpu'}
)

vectorstore = Chroma(
    persist_directory=str(DB_DIR), 
    embedding_function=embedding_model
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
all_docs = vectorstore.similarity_search("", k=10000)
bm25_retriever = BM25Retriever.from_documents(all_docs, k=5)

hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3] 
)

reranker_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=reranker_model, top_n=3)

final_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid_retriever
)

def format_docs(docs):
    return "\n\n".join(
        f"Content: {doc.page_content}\n(Page: {doc.metadata.get('page', 'Unknown')})" 
        for doc in docs
    )

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def llm_function(inputs: dict) -> str:
    context = inputs["context"]
    question = inputs["question"]
    
    prompt = f"""You are a trusted mental health assistant. Answer based EXCLUSIVELY on the provided context.

IMPORTANT RULES:
1. Answer in the EXACT SAME LANGUAGE as the question (Arabic or English).
2. If the answer is NOT in the context, you MUST say EXACTLY: "I do not have enough information in the document to answer this question." (or "لا أملك معلومات كافية في المستند للإجابة على هذا السؤال." if the question is in Arabic). 
3. CRITICAL: If you use the phrase in Rule 2, DO NOT add any page number or source citation.
4. If the answer IS in the context, you MUST mention the page number of the source in parentheses at the very end, for example: (Source: page 5) or (المصدر: الصفحة 5).

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024
    )
    
    answer = response.choices[0].message.content
    
    if not answer or answer.strip() == "":
        if any('\u0600' <= c <= '\u06FF' for c in question): # Check if question is Arabic
            return "لا أملك معلومات كافية في المستند للإجابة على هذا السؤال."
        else:
            return "I do not have enough information in the document to answer this question."
            
    return answer

rag_chain = (
    {"context": final_retriever | format_docs, "question": RunnablePassthrough()}
    | RunnableLambda(llm_function)
)

if __name__ == "__main__":
    test_questions = [
        "ما هو اضطراب القلق العام GAD؟",
        "What is PTSD and how is it diagnosed?",
        "كيف يمكن للمدير أن يدعم موظفيه نفسياً؟",
    ]
    
    print("="*70)
    print("🔍 Testing OPTIMIZED HYBRID SEARCH (70/30) + RE-RANKER + GROQ")
    print("="*70)
    
    for q in test_questions:
        print(f"\n❓ Question: {q}")
        try:
            retrieved_docs = final_retriever.invoke(q)
            print(f"📚 Retrieved {len(retrieved_docs)} chunks:")
            for i, doc in enumerate(retrieved_docs, 1):
                print(f"   {i}. Page {doc.metadata.get('page', '?')}: {doc.page_content[:60]}...")
            
            answer = rag_chain.invoke(q)
            print(f"\n💡 Answer:\n{answer}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}\n")
        print("-" * 70)