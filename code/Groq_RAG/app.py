import streamlit as st
import os
import time
import asyncio
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("Groq_Api_Key")
if not groq_api_key:
    st.error("GROQ API key not found. Please check your .env file or environment variables.")
    st.stop()

# Initialize LLM with Groq API
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define prompt template
prompt = ChatPromptTemplate.from_template("""
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
""")

# Initialize session state variables if they don't exist
for key in ["vectors", "embeddings", "loader", "docs", "text_splitter", "final_documents"]:
    if key not in st.session_state:
        st.session_state[key] = None

def create_vector_embedding():
    if "vectors" in st.session_state and st.session_state.vectors is not None:
        st.write("Vector store already initialized.")
        return
    
    try:
        st.session_state.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        
        if not os.path.exists("research_papers"):
            st.error("The 'research_papers' directory does not exist. Please create it and add PDFs.")
            return
        
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()

        if not st.session_state.docs:
            st.error("No documents found during the loading process.")
            return
        
        st.write(f"Loaded {len(st.session_state.docs)} documents.")

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        
        if not st.session_state.final_documents:
            st.error("Document splitting failed or produced empty results.")
            return
        
        st.write(f"Created {len(st.session_state.final_documents)} document chunks.")

        # Create the FAISS vector store
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

        # Verify vector store creation
        if isinstance(st.session_state.vectors, FAISS):
            st.write("Vector store initialized correctly.")
        else:
            st.error("Failed to initialize the FAISS vector store.")
            st.session_state.vectors = None  # Explicitly set to None to prevent retrieval issues
    except Exception as e:
        st.error(f"Error during vector embedding creation: {str(e)}")

# Streamlit UI components
st.title("RAG Document Q&A With Groq and Ollama")

user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

# Ensure vector store is initialized before querying
async def async_invoke(chain, query):
    return await chain.ainvoke({'input': query})

if user_prompt:
    if "vectors" not in st.session_state or st.session_state.vectors is None:
        st.error("Vectors not initialized. Please create the vector database first.")
    else:
        try:
            document_chain = create_stuff_documents_chain(llm, prompt)
            
            # Check if FAISS vector store supports retrieval
            if isinstance(st.session_state.vectors, FAISS):  # Ensure it's a FAISS instance
                retriever = st.session_state.vectors.as_retriever()
            else:
                st.error("Vector store is not properly initialized or does not support retrieval.")
                st.stop()
            
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            start = time.process_time()
            
            # Run async function in sync mode using asyncio
            response = asyncio.run(async_invoke(retrieval_chain, user_prompt))
            
            st.write(f"Response time: {time.process_time() - start} seconds")
            
            if "answer" in response:
                st.write(response['answer'])
            else:
                st.error("No valid response received from the retrieval chain.")
            
            if "context" in response:
                with st.expander("Document Similarity Search"):
                    for i, doc in enumerate(response['context']):
                        st.write(doc.page_content)
                        st.write('------------------------')
            else:
                st.warning("No context retrieved.")
        except Exception as e:
            st.error(f"Error during retrieval: {str(e)}")