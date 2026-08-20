from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

DATA_DIR = Path("data/sample_documents")
QUERY = "Why should sensitive information use local AI systems?"
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

def load_texts():
    docs = []
    for file_path in DATA_DIR.glob("*.txt"):
        docs.append({
            "source": file_path.name,
            "text": file_path.read_text(encoding="utf-8")
        })
    return docs

def test_config(chunk_size, overlap):
    client = chromadb.Client()
    collection = client.create_collection(
        name=f"test_{chunk_size}_{overlap}",
        metadata={"hnsw:space": "cosine"}
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for doc in load_texts():
        splits = splitter.split_text(doc["text"])
        for i, text in enumerate(splits):
            chunks.append({
                "id": f"{doc['source']}-{i}",
                "source": doc["source"],
                "text": text
            })

    embeddings = model.encode(
        [c["text"] for c in chunks],
        normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[{"source": c["source"]} for c in chunks]
    )

    query_embedding = model.encode(
        [QUERY],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(3, len(chunks)),
        include=["documents", "distances", "metadatas"]
    )

    print("\n" + "=" * 80)
    print(f"CHUNK SIZE: {chunk_size} | OVERLAP: {overlap}")
    print(f"TOTAL CHUNKS: {len(chunks)}")

    for rank, (doc, dist, meta) in enumerate(
        zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]
        ),
        start=1
    ):
        print(f"\nRank {rank} | Source: {meta['source']} | Distance: {dist:.4f}")
        print(doc[:350].replace("\n", " "))

configs = [
    (300, 50),
    (800, 150),
    (1500, 200),
]

for chunk_size, overlap in configs:
    test_config(chunk_size, overlap)
