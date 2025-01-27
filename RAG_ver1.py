# Version1 RetrievalQA
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
USER_AGENT = os.getenv("USER_AGENT")

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="langchain_search_api_key_openai", type="password"
    )
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/2_Chat_with_search.py)"


st.title("🔎 LangChain - NFL Search")


# Initialize session state for storing messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]


# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# Construct the vector database
web_loader = WebBaseLoader([
    "https://www.nfl.com/news/all-news",
    "https://www.nfl.com/news/sza-to-join-kendrick-lamar-as-guest-during-super-bowl-halftime-performance"
    ]
)

data = web_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, 
    chunk_overlap = 0
)

all_splits = text_splitter.split_documents(data)

vectorstore = Chroma.from_documents(
    documents=all_splits, 
    embedding=OpenAIEmbeddings(),
    collection_name= "handbook",
    persist_directory= "rag_db"
    )

template = hub.pull("rlm/rag-prompt")


# Get question by the user
if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Create RAG chat bot
    os.environ["OPENAI_API_KEY"] = openai_api_key
    model = ChatOpenAI(model_name="gpt-4", temperature=0)

    qa_chain = RetrievalQA.from_llm(
        llm=model, 
        retriever=vectorstore.as_retriever(),
        prompt = template
    )

    result = qa_chain.invoke({"query": prompt})

    msg = result["result"]

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)