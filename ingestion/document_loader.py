from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path("data/sample_documents")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def load_documents():
    documents = []

    for file_path in DATA_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for document in documents:
        split_texts = splitter.split_text(document["text"])

        for index, text in enumerate(split_texts):
            chunks.append({
                "id": f"{document['source']}-{index}",
                "source": document["source"],
                "text": text
            })

    return chunks


if __name__ == "__main__":
    documents = load_documents()
    chunks = chunk_documents(documents)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    for chunk in chunks:
        print("\n" + "=" * 70)
        print(f"Chunk ID: {chunk['id']}")
        print(f"Source: {chunk['source']}")
        print(f"Characters: {len(chunk['text'])}")
        print("-" * 70)
        print(chunk["text"])
