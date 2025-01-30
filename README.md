# RAG_practice

## Overview
NFL RAG is a Retrieval-Augmented Generation (RAG) model designed to provide real-time NFL news updates without requiring users to manually search for the latest information. The model extracts news links from the official NFL news page, builds a vector database using the collected articles, and generates the most relevant responses based on user queries while maintaining conversational context.

## Features
- **Real-Time News Extraction**: Automatically retrieves the latest news articles from the NFL official news page.
- **Vector Database Storage**: Utilizes ChromaDB to store and index extracted articles for efficient retrieval.
- **Conversational Memory**: Utilizes LangGraph to remember the user's previous interactions, allowing for more natural and contextual conversations.


## Technical Details
### webutil.py
- Contains the **crawl** function, which is responsible for parsing news links from the official NFL news page.
- Utilizes the **BeautifulSoup library** to extract and process the most recent articles from the webpage.

### RAG.py
- Builds a vector database using **LangChain's Chroma** to store and retrieve news articles efficiently.
- Implements a RAG model with **LangGraph** to maintain conversational context and improve response relevance over time.

## Requirements
To run NFL RAG, ensure you have all the libraries included in `requirements.txt` installed:
`pip install -r requirements.txt`

## Usage
1. Type `streamlit run RAG.py` in your command window to start the RAG model.
2. Provide your OpenAI API Key in the sidebar of Streamlit interface.
2. The vector database containing 32 most recent NFL news will be constructed in rag_db directory.
3. Interact with RAG using natural language queries through Streamlit interface.