import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from config import *

# Initialize embeddings model
embed_fn = HuggingFaceEmbeddings(model_name=embedding_model)

# Function to query Pinecone index and retrieve results
def query_to_vector_db(api_key, Index_name, embedding_model, query):
    query_vector = embedding_model.embed_query(query)
    pc = Pinecone(api_key=api_key)
    index = pc.Index(Index_name)
    result = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True  # Include metadata to get the text
    )
    return [match['metadata']['text'] for match in result['matches']]

# Function to get LLM model
def get_llm_model(model_name):
    llm = Ollama(model=model_name)
    return llm

# Function for semantic search using RAG
def semantic_search_rag(retriever, query, llm_chain_model):
    template = '''Give the answer of Statement from the following piece of context:
    context: {context} 
    statement: {statement}
    '''
    prompt = ChatPromptTemplate.from_template(template)

    if isinstance(retriever, list):
        retriever = " ".join(retriever)

    setup_and_retrieval = RunnableParallel(
        {"context": RunnablePassthrough(), "statement": RunnablePassthrough()}
    )

    output_parser = StrOutputParser()

    context = setup_and_retrieval.invoke({"context": retriever, "statement": query})
    prompt_answer = prompt.invoke(context)
    model_answer = llm_chain_model.invoke(prompt_answer)
    response = output_parser.invoke(model_answer)

    return response

# Streamlit UI
def main():
    # Set background image
    st.markdown(
        """
        <style>
        .reportview-container {
            background: url('\lhr.jpg');
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("FAST NUCES Chatbot")

    # Semantic search options
    query = st.text_input("Enter your query")
    if st.button("Search"):
        result = query_to_vector_db(api_key, Index_name, embed_fn, query)
        llm_chain_model = get_llm_model(model_name="phi3")
        output = semantic_search_rag(result, query, llm_chain_model)
        st.write("Response:")
        st.write(output)

if __name__ == "__main__":
    main()
