"""
Builds a merged retriever across all three Chroma collections:
  - faq     : FAQ entries
  - tickets : Resolved support tickets
  - guides  : Telecom guide chunks
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

CHROMA_DIR = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Lower score = Better match
SIMILARITY_THRESHOLD = 1.0


def build_retriever(
    k_faq: int = 3,
    k_tickets: int = 3,
    k_guides: int = 3,
) -> RunnableLambda:

    # Load embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL
    )

    # Open Chroma collections
    faq_store = Chroma(
        collection_name="faq",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    tickets_store = Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    guides_store = Chroma(
        collection_name="guides",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # Custom retrieval function
    def retrieve(query: str) -> list[Document]:

        all_docs = []

        stores = [
            (faq_store, k_faq),
            (tickets_store, k_tickets),
            (guides_store, k_guides),
        ]

        for store, k in stores:

            results = store.similarity_search_with_score(
                query=query,
                k=k,
            )

            for doc, score in results:

                # Store similarity score
                doc.metadata["score"] = round(score, 4)

                all_docs.append(doc)

        # Sort by relevance
        all_docs.sort(
            key=lambda doc: doc.metadata["score"]
        )

        # Keep only good matches
        filtered_docs = [
            doc
            for doc in all_docs
            if doc.metadata["score"] <= SIMILARITY_THRESHOLD
        ]

        return filtered_docs

    return RunnableLambda(retrieve)