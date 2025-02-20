import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
USER_AGENT = os.getenv("USER_AGENT")

from langchain_openai import ChatOpenAI
from langchain import hub

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import database

with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="langchain_search_api_key_openai", type="password"
    )
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/JHshin6688/RAG_practice)"
st.title("🏈 NFL RAG")

# Initialize session state for storing messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a NFL chatbot. How can I help you?"}
    ]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Get question by the user
if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Initialize vector database with the most recent news articles
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = database.create_database("https://www.nfl.com/news/all-news")

    vectorstore = st.session_state["vectorstore"]
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Create RAG chain
    os.environ["OPENAI_API_KEY"] = openai_api_key
    model = ChatOpenAI(model_name="gpt-4", temperature=0)
    
    combine_docs_chain = create_stuff_documents_chain(model, retrieval_qa_chat_prompt)
    qa_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

    result = qa_chain.invoke({"input": prompt})
    msg = result["answer"]

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)