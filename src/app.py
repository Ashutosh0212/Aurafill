"""Main Streamlit application for the chat interface.

This module implements the main Streamlit interface for the chat application,
including vector database selection, text input/upload, and query processing.
"""

import streamlit as st
from pathlib import Path
import tempfile
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import queue

from src.utils.embeddings import EmbeddingManager
from src.utils.db_utils import get_vectordb, get_available_databases
from src.utils.llm_utils import LLMManager
from src.config.config import QUERY_MODELS, TOP_K_RESULTS

# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None
    if "source_documents" not in st.session_state:
        st.session_state.source_documents = []
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "current_result" not in st.session_state:
        st.session_state.current_result = None

# Initialize session state
init_session_state()

# Initialize managers
embedding_manager = EmbeddingManager()
llm_manager = LLMManager()

async def process_query_async(query: str) -> Dict[str, Any]:
    """Process query asynchronously."""
    if not st.session_state.chain:
        st.error("Please load a database first!")
        return None
    
    try:
        result = await llm_manager.process_query(st.session_state.chain, query)
        return result
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None

# Page configuration
st.set_page_config(page_title="Chat with Documents", layout="wide")
st.title("Chat with Documents")

# Sidebar for database and model selection
with st.sidebar:
    st.header("Settings")
    
    # Database operations mode
    operation_mode = st.radio(
        "Operation Mode",
        ["Create New Database", "Query Existing Database"],
        index=1
    )
    
    if operation_mode == "Create New Database":
        # New database name input
        new_db_name = st.text_input(
            "Enter Database Name",
            placeholder="e.g., lecture_7"
        )
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=["txt", "pdf", "doc", "docx"]
        )
        
        # Text input
        text_input = st.text_area(
            "Or paste text directly",
            height=200
        )
        
        if st.button("Create Database"):
            if not new_db_name:
                st.error("Please enter a database name.")
                st.stop()
                
            if not uploaded_file and not text_input:
                st.error("Please provide either a file or text input.")
                st.stop()
            
            with st.spinner("Processing documents..."):
                if uploaded_file:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp_file.name)
                    
                    with open(tmp_path, "r", encoding="utf-8") as f:
                        text_content = f.read()
                    tmp_path.unlink()  # Clean up temporary file
                else:
                    text_content = text_input
                
                # Initialize vector database
                vectordb = get_vectordb(new_db_name, embedding_manager.embeddings)
                
                # Store documents
                vectordb.store_documents([text_content])
                st.session_state.vectordb = vectordb
            
            st.success(f"Database '{new_db_name}' created successfully!")
    
    else:  # Query Existing Database
        # Get available databases
        available_dbs = get_available_databases()
        
        if not available_dbs:
            st.warning("No existing databases found. Please create a new database first.")
            st.stop()
        
        # Database selection
        selected_db = st.selectbox(
            "Select Database",
            options=available_dbs
        )
        
        # Model selection
        selected_model = st.selectbox(
            "Select Query Model",
            options=QUERY_MODELS,
            index=0
        )
        
        if st.button("Load Database"):
            with st.spinner("Loading database..."):
                # Load the selected database
                vectordb = get_vectordb(selected_db, embedding_manager.embeddings)
                st.session_state.vectordb = vectordb
                
                # Create query processing setup
                st.session_state.chain = llm_manager.create_chain(
                    vectordb,
                    model_name=selected_model
                )
                
                # Clear previous messages and documents
                st.session_state.messages = []
                st.session_state.source_documents = []
            
            st.success(f"Database '{selected_db}' loaded successfully!")

# Main chat interface
st.divider()

# Query input
query = st.chat_input("Enter your question")

if query and not st.session_state.processing:
    st.session_state.processing = True
    
    try:
        if st.session_state.vectordb is None:
            st.error("Please load a database first!")
            st.stop()
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Process query asynchronously
        with st.spinner("Processing query..."):
            result = asyncio.run(process_query_async(query))
            st.session_state.current_result = result  # Store result in session state
            
            if result:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"]
                })
                st.session_state.source_documents = result.get("source_documents", [])
    
    finally:
        st.session_state.processing = False

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Display extracted content in an expander
if "source_documents" in st.session_state:
    del st.session_state.source_documents  # Remove old source_documents from state

if st.session_state.current_result and st.session_state.current_result.get("extracted_content"):
    with st.expander(f"Extracted Content (from {st.session_state.current_result.get('source_count', 0)} sources)"):
        for i, content in enumerate(st.session_state.current_result["extracted_content"], 1):
            st.markdown("**Extracted Information:**")
            st.markdown(content["content"])
            if "relevance" in content:
                st.markdown(f"*{content['relevance']}*")
            if i < len(st.session_state.current_result["extracted_content"]):
                st.markdown("---") 