# RAG_practice

## Overview
NFL RAG is a Retrieval-Augmented Generation (RAG) model designed to provide real-time NFL news updates without requiring users to manually search for the latest information. The model extracts news links from the official NFL news page, builds a vector database using the collected articles, and generates the most relevant responses based on user queries while maintaining conversational context.

## Features
- **Real-Time News Extraction**: Automatically retrieves the latest news articles from the NFL official news page.
- **Vector Database Storage**: Utilizes ChromaDB to store and index extracted articles for efficient retrieval.
- **Conversational Memory**: Utilizes LangGraph to remember the user's previous interactions, allowing for more natural and contextual conversations.


## Technical Details
### webutil.py
- Utilizes the **BeautifulSoup** library to extract and process the most recent articles from the webpage.
- Contains the **crawl** function, which is responsible for parsing news links from the official NFL news page.

### database.py
- Builds a vector database using **LangChain's Chroma** to store and retrieve news articles efficiently.
- The vector database will be constructed in the **vector_database** directory.

### RAG.py
- Implements a RAG model with **LangGraph** to maintain conversational context and improve response relevance over time.
- Utilizes **Streamlit** to build a RAG interface.

## Requirements
To run NFL RAG, ensure you have all the libraries included in `requirements.txt` installed:
`pip install -r requirements.txt`
Also, you have to create an environment file that contains your LangSmith API key.

## Usage
0. Feel free to delete the outdated 'vector_database' directory after you clone the project.  
1. Run `streamlit run RAG.py` command to start the RAG model.
2. Provide your OpenAI API Key in the sidebar of Streamlit interface.
3. Interact with RAG using natural language queries through Streamlit interface.
