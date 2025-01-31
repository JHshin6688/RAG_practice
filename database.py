from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import webutil
import os
import shutil

def create_database(url):
    # Crawl the news from the official website
    links = webutil.crawl(url)
    print(links)

    # Construct the vector database
    web_loader = WebBaseLoader(links)
    data = web_loader.load()

    text_splitter = TokenTextSplitter(chunk_size = 500, chunk_overlap = 0)
    all_splits = text_splitter.split_documents(data)

    dir_path = "vector_database"

    if os.path.exists(dir_path):
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

    vectorstore = Chroma.from_documents(
        documents=all_splits, 
        embedding=OpenAIEmbeddings(),
        collection_name= "handbook",
        persist_directory= dir_path
        )

    return vectorstore