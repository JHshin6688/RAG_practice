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

prompt = hub.pull("rlm/rag-prompt")

model = ChatOpenAI(model_name="gpt-4", temperature=0)

qa_chain = RetrievalQA.from_llm(
    llm=model, 
    retriever=vectorstore.as_retriever(),
    prompt = prompt
)

question1 = "What is the relationship between SZA and Kendrick Lamar?"
result1 = qa_chain.invoke({"query": question1})
print(result1["result"])