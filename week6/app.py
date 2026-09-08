import streamlit as st
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

st.set_page_config(
    page_title="Mental Health RAG Assistant",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Mental Health Q&A Assistant")
st.markdown("*Grounded RAG System based on the Saudi MoH Mental Health Guide*")
st.markdown("---")

st.sidebar.header("⚙️ Retrieval Settings")

comparison_mode = st.sidebar.checkbox(
    "🔬 Comparison Mode",
    value=False,
    help="Enable to manually switch between retrieval methods. "
         "When off, the app automatically uses the best-performing pipeline.",
)

if comparison_mode:
    retrieval_method = st.sidebar.selectbox(
        "Select Retrieval Method",
        [
            "Baseline (Vector Only)",
            "With Re-ranker",
            "Hybrid Search (70% Vector + 30% BM25) 🏆",
        ],
        index=2, 
    )
else:
    retrieval_method = "Hybrid Search (70% Vector + 30% BM25) 🏆"
    st.sidebar.success(
        "🏆 **Auto mode:** Using Hybrid Search "
        "(best Refusal Accuracy: 85.7% in Lab 6.4 evaluation)."
    )

st.sidebar.markdown("---")
st.sidebar.info(
    "**Design Choice:** Hybrid Search uses a 70/30 weight split because BM25 (keyword search) "
    "is weaker with Arabic morphology, while Vector Search captures semantic meaning better."
)

@st.cache_resource
def load_components():
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small", 
        model_kwargs={'device': 'cpu'}
    )
    
    if not DB_DIR.exists():
        st.error(f"❌ Database not found at: {DB_DIR}\nPlease run `ingest.py` first!")
        st.stop()
        
    vectorstore = Chroma(persist_directory=str(DB_DIR), embedding_function=embedding_model)
    all_docs = vectorstore.similarity_search("", k=10000)
    
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)
    
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    return vectorstore, all_docs, reranker, groq_client

vectorstore, all_docs, reranker, groq_client = load_components()

def get_retriever(method):
    if method == "Baseline (Vector Only)":
        return vectorstore.as_retriever(search_kwargs={"k": 3})
        
    elif method == "With Re-ranker":
        base = vectorstore.as_retriever(search_kwargs={"k": 10})
        return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base)
        
    elif method == "Hybrid Search (70% Vector + 30% BM25) 🏆":
        vec_ret = vectorstore.as_retriever(search_kwargs={"k": 5})
        bm25_ret = BM25Retriever.from_documents(all_docs, k=5)
        hybrid = EnsembleRetriever(retrievers=[vec_ret, bm25_ret], weights=[0.7, 0.3])
        return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=hybrid)

retriever = get_retriever(retrieval_method)

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

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024 
    )
    
    answer = response.choices[0].message.content
    
    if not answer or answer.strip() == "":
        if any('\u0600' <= c <= '\u06FF' for c in question):
            return "لا أملك معلومات كافية في المستند للإجابة على هذا السؤال."
        return "I do not have enough information in the document to answer this question."
        
    return answer

def format_docs(docs):
    return "\n\n".join(
        f"Content: {doc.page_content}\n(Page: {doc.metadata.get('page', 'Unknown')})" 
        for doc in docs
    )

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RunnableLambda(llm_function)
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📚 View Sources / عرض المصادر"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {i}** (Page {src['page']}): {src['content'][:200]}...")

if prompt := st.chat_input("Ask a question about mental health... / اسأل سؤالاً..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking... / جاري التفكير..."):
            retrieved_docs = retriever.invoke(prompt)
            answer = rag_chain.invoke(prompt)
            st.markdown(answer)
            
            sources = [
                {"page": doc.metadata.get("page", "?"), "content": doc.page_content} 
                for doc in retrieved_docs
            ]
            with st.expander("📚 View Sources / عرض المصادر"):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**Source {i}** (Page {src['page']}): {src['content'][:250]}...")
                    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer, 
        "sources": sources
    })

st.markdown("---")
st.caption("⚠️ *This assistant is for informational purposes only and does not replace professional medical advice.*")