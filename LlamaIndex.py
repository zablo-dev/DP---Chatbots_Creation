import os
from dotenv import load_dotenv
import openai
import chromadb
from flask import Flask, render_template, request

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


DATA_DIR = "data"
CHROMA_DIR = "chroma_db"


chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
chroma_collection = chroma_client.get_or_create_collection("pdf_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection, embedding_function=None)


Settings.llm = OpenAI(model="gpt-4")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


documents = SimpleDirectoryReader(DATA_DIR).load_data()
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
query_engine = index.as_query_engine(similarity_top_k=3)

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def chat():
    answer = ""
    if request.method == "POST":
        user_input = request.form["question"]
        if user_input.strip():
            response = query_engine.query(user_input)
            answer = str(response)
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
