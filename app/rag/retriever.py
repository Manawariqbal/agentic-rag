from app.config import settings


class Retriever:
    def __init__(
        self,
        vector_store,
        embedding_provider=None,
        top_k=None,
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

        self.top_k = (
            top_k
            if top_k is not None
            else settings.retrieval_top_k
        )

    def retrieve(self, query: str):
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        results = self.vector_store.search(
            query=query,
            top_k=self.top_k,
        )

        return [
            {
                "text": result["text"],
                "score": result["score"],
                "metadata": result["metadata"],
            }
            for result in results
        ]