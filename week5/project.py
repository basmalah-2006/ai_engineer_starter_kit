import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load and Clean Data
df = pd.read_csv("week5/ProductsData.csv", encoding="latin1")

# Drop rows with missing product names
df = df.dropna(subset=["Product_name"])

# Clean columns from extra quotation marks
for col in df.columns:
    df[col] = df[col].astype(str).str.replace('"', '').str.strip()

# Combine product information into a single text for embedding
# (name + category + region = richer context for search)
df["full_text"] = (
    df["Product_name"] + " | " + 
    df["Product_Category"] + " | " + 
    df["Region_address"] + " " + 
    df["Local_address"]
)

print(f"✅ Loaded {len(df)} products")
print("Example of combined text:")
print(df["full_text"].iloc[0])
print("-" * 50)

# 2. Load Multilingual Embedding Model
# Using multilingual model because data contains French + Arabic + English
model_name = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)
print(f"✅ Loaded model: {model_name}")

# 3. Create Chroma Database and Store Vectors
client = chromadb.Client()  # In-memory database (local)

# Delete collection if it already exists
try:
    client.delete_collection("products_collection")
except:
    pass

collection = client.create_collection(
    name="products_collection",
    metadata={"description": "Moroccan products marketplace"}
)

# Generate embeddings for all texts
texts = df["full_text"].tolist()
embeddings = model.encode(texts, show_progress_bar=True)

# Prepare metadata (additional information for each product)
metadatas = []
for _, row in df.iterrows():
    metadatas.append({
        "product_id": row["Product_id"],
        "category": row["Product_Category"],
        "price": row["price"] if row["price"] != "nan" else "0",
        "region": row["Region_address"],
        "city": row["Local_address"],
        "seller_type": row["Professional_Publication"]
    })

# Save everything to Chroma
collection.add(
    ids=[str(i) for i in range(len(texts))],
    embeddings=embeddings.tolist(),
    documents=texts,
    metadatas=metadatas
)
print(f"✅ Stored {len(texts)} products in ChromaDB")
print("-" * 50)

# 4. Semantic Search Function
def semantic_search(query, top_k=5, category_filter=None, max_price=None):
    """
    Performs semantic search on products with optional filtering
    """
    # Convert query to vector
    query_embedding = model.encode([query]).tolist()
    
    # Build filter if exists
    where_filter = None
    conditions = []
    if category_filter:
        conditions.append({"category": {"$eq": category_filter}})
    if max_price:
        conditions.append({"price": {"$lte": str(max_price)}})
    
    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}
    
    # Perform search
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter
    )
    
    return results

# 5. Search Experiments (Try them!)
print("\n🔍 Experiment 1: Search for 'iPhone phone'")
results = semantic_search("iphone apple telephone", top_k=3)
for i, doc in enumerate(results["documents"][0]):
    print(f"{i+1}. {doc}")
    print(f"   📍 {results['metadatas'][0][i]['city']} | 💰 {results['metadatas'][0][i]['price']} DH")
print()

print("🔍 Experiment 2: Search for 'apartment for sale in Casablanca' (with filter)")
results = semantic_search(
    "appartement casablanca", 
    top_k=3,
    category_filter="Appartements "
)
for i, doc in enumerate(results["documents"][0]):
    print(f"{i+1}. {doc}")
    print(f"   💰 {results['metadatas'][0][i]['price']} DH")
print()

print("🔍 Experiment 3: Semantic search for 'powerful gaming laptop' (notice it will find PC Gamer)")
results = semantic_search("gaming laptop powerful", top_k=3)
for i, doc in enumerate(results["documents"][0]):
    print(f"{i+1}. {doc}")
print()

print("🔍 Experiment 4: Search for 'economic car' with max price 80000 DH")
results = semantic_search(
    "voiture pas cher economique", 
    top_k=3,
    max_price=80000
)
for i, doc in enumerate(results["documents"][0]):
    print(f"{i+1}. {doc}")
    print(f"   💰 {results['metadatas'][0][i]['price']} DH")