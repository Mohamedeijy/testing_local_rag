from langchain_nomic.embeddings import NomicEmbeddings, Embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.llms import Ollama, BaseLLM


def get_model() -> BaseLLM:
    return Ollama(model="llama3:8b",
                  keep_alive=1,  # keep model loaded to gain time
                  temperature=0,
                  )


def get_retriever(embed_model: Embeddings, top_k: int = 5) -> VectorStoreRetriever:
    index_name: str = "markdown-notes"
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embed_model,
    )
    return vectorstore.as_retriever(k=top_k)


def get_embed_model(dimensionality: int = 768) -> Embeddings:
    return NomicEmbeddings(model="nomic-embed-text-v1.5", dimensionality=dimensionality)
