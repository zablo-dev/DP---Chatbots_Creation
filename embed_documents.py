import os
import fitz
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from tqdm import tqdm

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection("pdf_chunks")

def extract_chunks_from_pdf(file_path):
    doc = fitz.open(file_path)
    chunks = []
    for page in doc:
        text = page.get_text()
        for para in text.split("\n\n"):
            if len(para.strip()) > 50:
                chunks.append(para.strip())
    return chunks

def embed_text(texts):
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return [e.embedding for e in response.data]

def index_pdfs(data_dir="data"):
    for filename in tqdm(os.listdir(data_dir)):
        if filename.endswith(".pdf"):
            path = os.path.join(data_dir, filename)
            print(f"📄 Indexing {filename}")
            chunks = extract_chunks_from_pdf(path)
            if not chunks:
                continue
            embeddings = embed_text(chunks)
            collection.add(
                documents=chunks,
                embeddings=embeddings,
                ids=[f"{filename}-{i}" for i in range(len(chunks))]
            )

if __name__ == "__main__":
    index_pdfs()
