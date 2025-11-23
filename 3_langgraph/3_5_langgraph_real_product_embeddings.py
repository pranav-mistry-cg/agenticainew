<<<<<<< HEAD
# https://www.kaggle.com/datasets/asaniczka/amazon-products-dataset-2023-1-4m-products?select=amazon_products.csv

=======
# Dataset: https://www.kaggle.com/code/jayrdixit/amazon-product-dataset/input?select=amazon_products.csv
>>>>>>> 843034120d515e626bea00ea96a943cc78c4d601
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from typing import TypedDict
import pandas as pd
import gradio as gr
import os

# --- Step 1: Define state ---
class ProductState(TypedDict):
    query: str
    results: str

<<<<<<< HEAD

# --- Step 2: Load dataset and embeddings ---
csv_path = "c://code//agenticai//3_langgraph//amazon_products.csv"

# Load the CSV (adjust column names if needed)
print("Loading product dataset...")
df = pd.read_csv(csv_path)
df = df.head(1000) # Only first 100 rows

# Keep only text and useful metadata
texts = df["title"].astype(str).tolist()
metadatas = df.to_dict(orient="records") # list of dicts, orient="records" means each row is a dict

# Create or load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Directory where Chroma will persist vectors
persist_dir = "c://code//agenticai//3_langgraph//product_embeddings_chroma"

# Create Chroma vector store if it doesn't exist yet
if not os.path.exists(persist_dir) or not os.listdir(persist_dir):
    print("Creating new Chroma vector store (first-time setup)...")
=======
# --- Step 2: Load dataset and embeddings ---
csv_path = "c://code//agenticai//3_langgraph//amazon_products.csv"
df = pd.read_csv(csv_path)
# df = df.head(1000)  # Limit for demo

texts = df["title"].astype(str).tolist()
metadatas = df.to_dict(orient="records")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Step 3: Setup Chroma vector store ---
persist_dir = "c://code//agenticai//3_langgraph//chromadb"
collection_name = "products_collection"

# Import chromadb client to check existing collections
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path=persist_dir, settings=Settings())

existing_collections = [col.name for col in client.list_collections()]

if collection_name in existing_collections:
    print(f"Loading existing collection '{collection_name}'...")
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name
    )
else:
    print(f"Creating new collection '{collection_name}'...")
>>>>>>> 843034120d515e626bea00ea96a943cc78c4d601
    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
<<<<<<< HEAD
        persist_directory=persist_dir,
    )
    vectordb.persist()
else:
    print("Loading existing Chroma vector store...")
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )


# --- Step 3: Define LangGraph nodes ---
def search_products(state: ProductState) -> ProductState:
    """Search similar products using Chroma."""
=======
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    vectordb.persist()

# --- Step 4: Define LangGraph nodes ---
def search_products(state: ProductState) -> ProductState:
>>>>>>> 843034120d515e626bea00ea96a943cc78c4d601
    results = vectordb.similarity_search(state["query"], k=3)
    titles = [doc.metadata.get("title", "Unknown Product") for doc in results]
    state["results"] = "\n".join([f"• {title}" for title in titles])
    return state

<<<<<<< HEAD

def format_response(state: ProductState) -> ProductState:
    """Format the response message."""
    if state["results"]:
        state["results"] = f"Found products:\n{state['results']}"
    else:
        state["results"] = "No products found."
    return state


# --- Step 4: Build the LangGraph ---
=======
def format_response(state: ProductState) -> ProductState:
    state["results"] = f"Found products:\n{state['results']}" if state["results"] else "No products found."
    return state

>>>>>>> 843034120d515e626bea00ea96a943cc78c4d601
graph = StateGraph(ProductState)
graph.add_node("search", search_products)
graph.add_node("format", format_response)
graph.set_entry_point("search")
graph.add_edge("search", "format")
graph.add_edge("format", END)
<<<<<<< HEAD

runnable = graph.compile()


# --- Step 5: Gradio handlers ---
def search(query):
    result = runnable.invoke({"query": query})
    return result["results"]

=======
runnable = graph.compile()

# --- Step 5: Gradio handlers ---
def search(query):
    return runnable.invoke({"query": query})["results"]
>>>>>>> 843034120d515e626bea00ea96a943cc78c4d601

def chat_fn(message, history):
    return search(message)

<<<<<<< HEAD

=======
>>>>>>> 843034120d515e626bea00ea96a943cc78c4d601
# --- Step 6: Gradio UI ---
demo = gr.ChatInterface(
    fn=chat_fn,
    title="Product Search (Chroma + HuggingFace)",
    examples=["wireless headphones", "gaming laptop", "DSLR camera"],
)

if __name__ == "__main__":
    demo.launch()
