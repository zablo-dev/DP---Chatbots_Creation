import os
import asyncio
from dotenv import load_dotenv
import chromadb
from openai import OpenAI

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole


load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)


chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection("pdf_chunks")




kernel = Kernel()
chat_service = OpenAIChatCompletion(
    service_id="chat-gpt",
    ai_model_id="gpt-4",
    api_key=openai_key
)
kernel.add_service(chat_service)


def get_context(query, k=3):
    embedding = client.embeddings.create(input=[query], model="text-embedding-3-small").data[0].embedding
    results = collection.query(query_embeddings=[embedding], n_results=k)
    return "\n".join(results["documents"][0])


async def main():
    print("🤖 Semantic Kernel RAG Chatbot (latest)")
    print("Zadej dotaz nebo 'exit' pro ukončení.\n")

    chat_history = []

    while True:
        query = input("Ty: ")
        if query.lower() in ["exit", "quit"]:
            break

        context = get_context(query)
        full_prompt = f"Na základě následujícího kontextu odpověz na otázku:\n{context}\n\nOtázka: {query}"

        chat_history.append(ChatMessageContent(role=AuthorRole.USER, content=full_prompt))
        response = await chat_service.complete_chat_async(chat_history)
        chat_history.append(response)

        print("Bot:", response.content)

if __name__ == "__main__":
    asyncio.run(main())





