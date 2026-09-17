import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

PDF_PATH = r"week7/data/CS212-Intro.-to-Artificial-Intelligence---All-Lecture-Notes-0-to-12-(@HNU)-Spring-2025 (3).pdf"
VECTOR_DB_PATH = "vector_store"


def extract_text_with_pages(pdf_path: str) -> list[dict]:
    """
    Extracts text from the PDF with page numbers for each part.
    Returns a list: [{"text": "...", "page": 1}, ...]
    """
    reader = PdfReader(pdf_path)
    pages_data = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip(): 
            pages_data.append({
                "text": text.strip(),
                "page": i + 1
            })
    
    print(f"✅ Extracted {len(pages_data)} pages with text")
    return pages_data


def chunk_pages(pages_data: list[dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[dict]:
    """
    Splits each page into chunks with overlap to avoid cutting sentences.
    Each chunk retains the original page number.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    
    chunks = []
    for page in pages_data:
        page_chunks = splitter.split_text(page["text"])
        for i, chunk in enumerate(page_chunks):
            chunks.append({
                "text": chunk,
                "page": page["page"],
                "chunk_index": i
            })
    
    print(f"✅ Split text into {len(chunks)} chunks")
    return chunks


def build_vector_store(chunks: list[dict]) -> FAISS:
    """
    Converts chunks to embeddings using a FREE local model and saves them in FAISS.
    """
    texts = [c["text"] for c in chunks]
    metadatas = [{"page": c["page"], "chunk_index": c["chunk_index"]} for c in chunks]
    
    print("⏳ Loading free embedding model (this may take a minute the first time)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("⏳ Building Vector Store...")
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    
    vectorstore.save_local(VECTOR_DB_PATH)
    print(f"✅ Vector Store saved to: {VECTOR_DB_PATH}")
    
    return vectorstore


def load_vector_store() -> FAISS:
    """Load the saved Vector Store"""
    if not os.path.exists(VECTOR_DB_PATH):
        raise FileNotFoundError("Vector Store not found. Run build_vector_store() first.")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(
        VECTOR_DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )


def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Retrieves the top_k most similar chunks to the query and returns them
    as formatted text with page numbers.
    """
    vectorstore = load_vector_store()
    docs = vectorstore.similarity_search(query, k=top_k)
    
    context = ""
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        context += f"\n--- [Source {i}: Page {page}] ---\n{doc.page_content}\n"
    
    return context


if __name__ == "__main__":
    pages = extract_text_with_pages(PDF_PATH)
    chunks = chunk_pages(pages)
    build_vector_store(chunks)
    
    print("\n🧪 Testing retrieval:")
    print(retrieve_context("What is A* search algorithm?"))