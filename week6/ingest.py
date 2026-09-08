import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

SCRIPT_DIR = Path(__file__).parent.resolve()

POSSIBLE_PDF_PATHS = [
    SCRIPT_DIR / "data" / "mental_health_guide.pdf",           
    SCRIPT_DIR / "week6" / "data" / "mental_health_guide.pdf", 
    Path("data") / "mental_health_guide.pdf",                  
    Path("week6") / "data" / "mental_health_guide.pdf",        
]

pdf_path = None
for path in POSSIBLE_PDF_PATHS:
    if path.exists():
        pdf_path = path
        break

if pdf_path is None:
    raise FileNotFoundError(
        f"❌ Could not find 'mental_health_guide.pdf' in any of these locations:\n"
        + "\n".join(f"  - {p}" for p in POSSIBLE_PDF_PATHS)
        + "\n\n💡 Please download the PDF and place it in one of the above folders."
    )

print(f"📄 Loading PDF from: {pdf_path}")

DB_DIR = SCRIPT_DIR / "chroma_db_multilingual"


loader = PyPDFLoader(str(pdf_path))
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", "، ", "؛ ", " ", ""],
    length_function=len,
)
chunks = text_splitter.split_documents(documents)


embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small",
    model_kwargs={'device': 'cpu'}
)


vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=str(DB_DIR)
)

print(f"✅ Document processed successfully!")
print(f"   📊 Chunks created: {len(chunks)}")
print(f"   💾 Database saved to: {DB_DIR}")