# Version3 RAG_ver2 + memory functionality
import streamlit as st
import os
import getpass
from dotenv import load_dotenv
load_dotenv()
USER_AGENT = os.getenv("USER_AGENT")

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import webutil

with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="langchain_search_api_key_openai", type="password"
    )
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

st.title("🏈 NFL RAG")


# Initialize session state for storing messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]


# Initialize session state for conversation memory
#if "memory" not in st.session_state:
#    st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Crawl the most recent 32 NFL news from the official website
initial_link = "https://www.nfl.com/news/all-news"
links = webutil.crawl(initial_link)

# Construct the vector database
web_loader = WebBaseLoader(links)
data = web_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
all_splits = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(
    documents=all_splits, 
    embedding=OpenAIEmbeddings(),
    collection_name= "handbook",
    persist_directory= "rag_db"
    )

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# Get question by the user
if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Create RAG chain
    os.environ["OPENAI_API_KEY"] = openai_api_key
    model = ChatOpenAI(model_name="gpt-4", temperature=0)

    combine_docs_chain = create_stuff_documents_chain(model, retrieval_qa_chat_prompt)
    qa_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

    result = qa_chain.invoke({"input": prompt})
    #print(result)
    msg = result["answer"]

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)