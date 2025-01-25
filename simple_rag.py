import os
from dotenv import load_dotenv
load_dotenv()
USER_AGENT = os.getenv("USER_AGENT")
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")

os.environ["OPENAI_API_KEY"] = openai_api_key

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

web_loader = WebBaseLoader([
    #"https://python.langchain.com/docs/get_started/introduction",   # LangChain Introduction
    #"https://python.langchain.com/docs/modules/data_connection/" # LangChain Retrieval
    "https://www.nba.com/news/power-rankings-2024-25-week-13"
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

prompt = hub.pull("rlm/rag-prompt")

model = ChatOpenAI(model_name="gpt-4", temperature=0)

qa_chain = RetrievalQA.from_llm(
    llm=model, 
    retriever=vectorstore.as_retriever(),
    prompt = prompt
)

question1 = "Which two teams have the highest power ranking in NBA?"
result1 = qa_chain.invoke({"query": question1})
print(result1["result"])

#question2 = "What is retrieval in LangChain?"
#result2 = qa_chain.invoke({"query" : question2})
#print(result2["result"])


'''
    with st.spinner("Processing.."):
        web_loader = WebBaseLoader(web_path= [web_link])

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

        model = ChatOpenAI(model_name="gpt-4", temperature=0)

        qa_chain = RetrievalQA.from_llm(
            llm=model, 
            retriever=vectorstore.as_retriever(),
            prompt = template
        )

        result = qa_chain.invoke({"query": web_link})
        st.session_state.messages.append({"role":"assistant", "content" : result["result"]})
        st.chat_message("assistant").write(result["result"])'''