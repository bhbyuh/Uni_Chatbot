# RAG Implementation with Pinecone on University Data

This repository demonstrates the implementation of a **Retrieval-Augmented Generation (RAG)** system using **Pinecone** as the vector database for storing and retrieving university data. The system scrapes the entire university website, stores the data in a Pinecone vector database, and provides a **Streamlit app** interface to interact with a **Phi-3 LLM** for querying the data.

## Tools Used
- **Pinecone**: A fully managed vector database used to store and retrieve embeddings from the scraped university website data.
- **Streamlit**: A framework for building interactive web applications, used to create the user interface for querying the university data.
- **Phi-3 LLM**: A language model used to generate responses based on the queries made by users through the Streamlit app.
- **Web Scraping Libraries**: (e.g., `requests`, `BeautifulSoup`) used to scrape the university website.
- **Langchain**: A framework for building RAG systems and integrating language models with vector databases (optional depending on your implementation).

## Workflow
1. **Scraping University Data**:
   - The first step involves scraping the entire university website using Python-based web scraping tools like `requests` and `BeautifulSoup`.
   - The scraped data includes various pages from the website, such as course listings, faculty details, news, etc.

2. **Embedding and Storing Data in Pinecone**:
   - After scraping, the textual content is transformed into embeddings using a pre-trained model.
   - These embeddings are then stored in **Pinecone**, a vector database designed for efficient similarity search and retrieval.

3. **Querying with Phi-3 LLM**:
   - The user can interact with the system by querying the website's content through a **Streamlit** interface.
   - The Phi-3 LLM processes the query, retrieves relevant information from the Pinecone vector database, and generates a response to answer the user's query.
