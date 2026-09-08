from pathlib import Path
from langchain_chroma import Chroma
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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
3. CRITICAL: If you use the phrase in Rule 2, DO NOT add any page number or source citation. NEVER.
4. If the answer IS in the context, you MUST mention the page number of the source in parentheses at the very end, for example: (Source: page 5) or (المصدر: الصفحة 5).

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",  
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024, 
    )
    answer = response.choices[0].message.content

    if not answer or answer.strip() == "":
        if any('\u0600' <= c <= '\u06FF' for c in question):
            return "لا أملك معلومات كافية في المستند للإجابة على هذا السؤال."
        return "I do not have enough information in the document to answer this question."

    return answer

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RunnableLambda(llm_function)
)

if __name__ == "__main__":
    query = "ما هي أهم عوامل الخطر التي تؤثر على الصحة النفسية في بيئة العمل؟"
    print("="*70)
    print(f"❓ Question: {query}")
    print("="*70)
    print("💡 Answer:")
    try:
        answer = rag_chain.invoke(query)
        print(answer)
    except Exception as e:
        print(f"❌ Error occurred: {e}")
    print("="*70)