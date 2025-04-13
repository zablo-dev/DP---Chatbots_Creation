import os
from dotenv import load_dotenv
import openai
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
import chromadb




load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


DATA_DIR = "data"
CHROMA_DIR = "chroma_db"


chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
chroma_collection = chroma_client.get_or_create_collection("pdf_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)


Settings.llm = OpenAI(model="gpt-4")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


documents = SimpleDirectoryReader(DATA_DIR).load_data()


index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store
)


query_engine = index.as_query_engine(similarity_top_k=3)


def chat():
    print(" FAQ-chatbot Annual meeting 2025")
    print("Zadej dotaz nebo napiš 'exit' pro ukončení.")

    while True:
        query = input("Ty: ")
        if query.lower() == "exit":
            print("Bot: Nashledanou!")
            break

        response = query_engine.query(query)
        print("Bot:", response)


if __name__ == "__main__":
    chat()
