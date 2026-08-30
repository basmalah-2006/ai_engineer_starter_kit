import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Load and Clean Data
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base_dir, "ProductsData.csv"), encoding="latin1")
    df = df.dropna(subset=["Product_name"])
    
    # Clean columns from extra quotation marks
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace('"', '').str.strip()
    
    # Combine product information into a single text for embedding
    df["full_text"] = (
        df["Product_name"] + " | " + 
        df["Product_Category"] + " | " + 
        df["Region_address"] + " " + 
        df["Local_address"]
    )
    return df

df = load_data()

# 2. Load Multilingual Embedding Model
@st.cache_resource
def load_model():
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    return SentenceTransformer(model_name)

model = load_model()

# 3. Create Chroma Database and Store Vectors
@st.cache_resource
def setup_chroma():
    client = chromadb.Client()
    
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
    
    # Prepare metadata
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
    
    # Save to Chroma
    collection.add(
        ids=[str(i) for i in range(len(texts))],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas
    )
    
    return collection

collection = setup_chroma()

# 4. Semantic Search Function
def semantic_search(query, top_k=5, category_filter=None, max_price=None):
    """
    Performs semantic search on products with optional filtering
    """
    query_embedding = model.encode([query]).tolist()
    
    # Build filter
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

# 5. Streamlit UI
st.title("🔍 Semantic Product Search Engine")
st.markdown("Search for products using natural language with semantic understanding")

# Search input
query = st.text_input("What are you looking for?", placeholder="e.g., iPhone phone, gaming laptop, apartment in Casablanca")

# Filters
col1, col2 = st.columns(2)

with col1:
    categories = ["All"] + sorted(df["Product_Category"].unique().tolist())
    category = st.selectbox("Filter by Category:", categories)

with col2:
    max_price = st.number_input("Max Price (DH)", min_value=0, value=0, step=1000)

# Search button
if st.button("🔎 Search", type="primary"):
    if query:
        with st.spinner("Searching..."):
            # Perform search
            category_filter = category if category != "All" else None
            price_filter = max_price if max_price > 0 else None
            
            results = semantic_search(
                query, 
                top_k=5,
                category_filter=category_filter,
                max_price=price_filter
            )
            
            # Display results
            if results["documents"][0]:
                st.success(f"Found {len(results['documents'][0])} results")
                st.markdown("---")
                
                for i, doc in enumerate(results["documents"][0], 1):
                    metadata = results["metadatas"][0][i-1]
                    
                    with st.expander(f"Result {i}: {doc.split('|')[0].strip()}", expanded=True):
                        st.markdown(f"**Full Description:** {doc}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📍 City", metadata["city"])
                        with col2:
                            st.metric("💰 Price", f"{metadata['price']} DH")
                        with col3:
                            st.metric("🏷️ Category", metadata["category"])
                        
                        st.caption(f"Region: {metadata['region']} | Seller: {metadata['seller_type']}")
                        st.markdown("---")
            else:
                st.warning("No results found. Try a different search query or adjust filters.")
    else:
        st.warning("Please enter a search query")

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This semantic search engine uses:
    - **Multilingual Embeddings** (supports Arabic, French, English)
    - **ChromaDB** for vector storage
    - **Cosine Similarity** for matching
    
    **Tips:**
    - Search in any language (Arabic, French, English)
    - Use natural language queries
    - Apply filters to narrow results
    """)
    
    st.header("📊 Dataset Info")
    st.metric("Total Products", len(df))
    st.metric("Categories", df["Product_Category"].nunique())
    st.metric("Regions", df["Region_address"].nunique())