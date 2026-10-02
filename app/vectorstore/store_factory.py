from app.config import VECTOR_STORE


def get_store():
    if VECTOR_STORE == "faiss":
        from app.vectorstore.faiss_store import FAISSStore
        return FAISSStore()
    raise ValueError(f"Unknown vector store: {VECTOR_STORE}")