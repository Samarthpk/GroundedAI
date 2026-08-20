import chromadb
from sentence_transformers import SentenceTransformer
from ingestion.document_loader import load_documents, chunk_documents

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "grounded_ai_docs"
TOP_K = 5

# Local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent local ChromaDB
client = chromadb.PersistentClient(path=CHROMA_PATH)


def build_vector_store():
    documents = load_documents()
    chunks = chunk_documents(documents)

    # Rebuild collection so rerunning does not create duplicates
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {"source": chunk["source"]}
            for chunk in chunks
        ]
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")
    return collection


def retrieve(query, top_k=TOP_K):
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        collection = build_vector_store()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    available = collection.count()
    k = min(top_k, available)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []

    for chunk_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved.append({
            "id": chunk_id,
            "text": document,
            "source": metadata["source"],
            "distance": distance
        })

    return retrieved


if __name__ == "__main__":
    build_vector_store()

    query = "Why should sensitive information use local AI systems?"

    print(f"\nQUERY: {query}")
    print("\nTOP RESULTS:")

    results = retrieve(query)

    for rank, result in enumerate(results, start=1):
        print("\n" + "=" * 70)
        print(f"Rank: {rank}")
        print(f"Chunk ID: {result['id']}")
        print(f"Source: {result['source']}")
        print(f"Cosine distance: {result['distance']:.4f}")
        print("-" * 70)
        print(result["text"])
