"""
PolyHedge - Conversational RAG over local Polymarket data
Uses: TF-IDF retrieval (instant, no GPU) + local flan-t5 LLM
Supports: JSON and CSV market data files
"""

import os
import json
import pandas as pd
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Use langchain_classic Document or fallback
try:
    from langchain_classic.schema import Document
except ImportError:
    from langchain_core.documents import Document

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = "Data"          # folder containing your JSON/CSV files
OLLAMA_MODEL = "llama3.1:8b"      # change to any model you have pulled e.g. mistral, gemma
EMBED_MODEL = "nomic-embed-text"  # ollama pull nomic-embed-text
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# ── 1. Load Data ──────────────────────────────────────────────────────────────

def load_json_file(path: str) -> list[Document]:
    """Load a JSON file — handles both list of records and single dict."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    docs = []
    for record in data:
        # Flatten the record into readable text so the LLM can reason over it
        text = "\n".join(f"{k}: {v}" for k, v in record.items())
        docs.append(Document(
            page_content=text,
            metadata={"source": path, "type": "json"}
        ))
    return docs


def load_csv_file(path: str) -> list[Document]:
    """Load a CSV file — sample large files, each row becomes a Document."""
    df = pd.read_csv(path)
    # Sample large CSVs to keep embedding fast
    if len(df) > 50:
        df = df.sample(n=50, random_state=42)
    docs = []
    for _, row in df.iterrows():
        text = "\n".join(f"{col}: {row[col]}" for col in df.columns)
        docs.append(Document(
            page_content=text,
            metadata={"source": path, "type": "csv"}
        ))
    return docs

def load_all_data(data_dir: str) -> list[Document]:
    """Recursively load all JSON and CSV files from the data directory."""
    all_docs = []
    path = Path(data_dir)

    json_files = list(path.glob("**/*.json"))
    csv_files = list(path.glob("**/*.csv"))

    print(f"📂 Found {len(json_files)} JSON file(s), {len(csv_files)} CSV file(s)")

    for f in json_files:
        try:
            docs = load_json_file(str(f))
            all_docs.extend(docs)
            print(f"  ✅ Loaded {str(f)} → {len(docs)} records")
        except Exception as e:
            print(f"  ❌ Failed to load {f}: {e}")

    for f in csv_files:
        try:
            docs = load_csv_file(str(f))
            all_docs.extend(docs)
            print(f"  ✅ Loaded {str(f)} → {len(docs)} rows")
        except Exception as e:
            print(f"  ❌ Failed to load {f}: {e}")

    return all_docs

# ── 2. Build Vector Store ─────────────────────────────────────────────────────

def build_vectorstore(docs: list[Document]):
    """Build a TF-IDF retriever — instant, no GPU needed."""
    from langchain_community.retrievers import TFIDFRetriever
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"  📄 {len(docs)} documents → {len(chunks)} chunks")
    
    retriever = TFIDFRetriever.from_documents(chunks, k=6)
    print("  ✅ TF-IDF retriever built")
    return retriever

  
# ── 3. Build RAG Chain ────────────────────────────────────────────────────────

def build_chain(retriever):
    """Wire up the RAG chain with TF-IDF retriever + local LLM."""
    from langchain_huggingface import HuggingFacePipeline
    from transformers import pipeline
    
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        device=-1,  # CPU
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    prompt = ChatPromptTemplate.from_template(
"""You are an expert analyst for Polymarket prediction market data.
Answer based on the data provided. Be specific and reference numbers.
If the data doesn't contain the answer, say so clearly.

Context:
{context}

Question: {question}""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ── 4. Conversational Loop ────────────────────────────────────────────────────
def print_sources(source_docs: list[Document]):
    """Print the source chunks the answer was grounded in."""
    seen = set()
    for doc in source_docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            print(f"    📎 {src}")
            seen.add(src)


def run_chat(chain):
    """Main conversational while loop."""
    print("\n" + "="*60)
    print("  PolyHedge AI — Ask anything about your market data")
    print("  Type 'exit' or 'quit' to stop | 'sources' to toggle source display")
    print("="*60 + "\n")

    show_sources = False

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        if user_input.lower() == "sources":
            show_sources = not show_sources
            print(f"  [Source display {'ON' if show_sources else 'OFF'}]\n")
            continue

        if user_input.lower() == "clear":
            chain.memory.clear()
            print("  [Conversation memory cleared]\n")
            continue

        try:
            result = chain.invoke(user_input)
            answer = str(result)
            print(f"\nPolyHedge: {answer}\n")

        except Exception as e:
            print(f"\n⚠️  Error: {e}\n")

def main():
    # Check data dir exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory '{DATA_DIR}' not found.")
        print("   Create it and drop your JSON/CSV Polymarket data files in.")
        return

    # Load → Embed → Chat
    docs = load_all_data(DATA_DIR)
    if not docs:
        print("❌ No documents loaded. Check your data directory.")
        return

    vectorstore = build_vectorstore(docs)
    chain = build_chain(vectorstore)
    run_chat(chain)


if __name__ == "__main__":
    main()

