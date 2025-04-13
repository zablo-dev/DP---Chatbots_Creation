import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import streamlit as st


load_dotenv()


DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"
NUM_RESULTS = 5




embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH
)
retriever = vector_store.as_retriever(search_kwargs={'k': NUM_RESULTS})


llm = ChatOpenAI(temperature=0.5, model='gpt-4o-mini')

# Streamlit UI
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 FAQ CHAT-bot GBS ANNUAL EVENT 2025")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


user_input = st.chat_input("Ask a question...")


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})


    docs = retriever.invoke(user_input)
    knowledge = "\n\n".join([doc.page_content for doc in docs])

    # RAG prompt
    prompt = f"""
You are an internal assistant answering frequently asked questions for a company event or service.
Use only the knowledge provided below to answer. Do not make up information.

If the question cannot be answered from the provided knowledge, respond: 
"I'm sorry, I couldn't find the answer in the document."

Include a short reference to the document if relevant (e.g., 'as stated in the document').

Question: {user_input}

Conversation history: {st.session_state.chat_history}

Knowledge base:
{knowledge}
"""



    response = llm.invoke(prompt)
    bot_reply = response.content


    with st.chat_message("assistant"):
        st.markdown(bot_reply)
    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
