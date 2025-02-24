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

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Class Chat Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.utils.embeddings import EmbeddingManager
from src.utils.db_utils import get_vectordb, get_available_databases
from src.utils.llm_utils import LLMManager
from src.config.config import QUERY_MODELS, TOP_K_RESULTS
from src.config.users import authenticate
from src.utils.user_logs import UserLogger

# Custom CSS for styling
st.markdown("""
<style>
    /* Dark theme */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 95%;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
        width: 300px;
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    /* Buttons */
    .stButton button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 6px;
    }
    
    /* Input fields */
    .stTextInput input, .stTextArea textarea {
        background-color: #1e293b;
        color: #e2e8f0;
        border: 1px solid #475569;
    }
    
    /* Chat interface styling */
    .chat-interface {
        display: flex;
        flex-direction: column;
        margin: -1rem;
        padding: 1rem;
    }
    
    .chat-container {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        overflow-y: auto;
    }
    
    .stChatMessage {
        background-color: #0f172a !important;
        margin: 0.5rem 0;
        padding: 0.75rem;
        border-radius: 6px;
    }

    /* Success/Error messages */
    .stSuccess, .stError {
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }

    /* Form styling */
    .stForm {
        background-color: #1e293b;
        padding: 1rem;
        border-radius: 8px;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1e293b;
        border-radius: 6px;
    }

    /* User info card */
    .user-info {
        background-color: #1e293b;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #475569;
    }

    /* Chat input container */
    .stChatInputContainer {
        padding: 1rem;
        background-color: #1e293b;
        border-radius: 8px;
        margin-top: 0.5rem;
        position: sticky;
        bottom: 0;
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)

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
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_role" not in st.session_state:
        st.session_state.user_role = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "user_logger" not in st.session_state:
        st.session_state.user_logger = None

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

# Login section
if not st.session_state.authenticated:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style='padding: 2rem;'>
                <h1 style='color: #e2e8f0; margin-bottom: 1.5rem;'>Class Chat Assistant</h1>
                <div style='background-color: #1e293b; padding: 2rem; border-radius: 8px;'>
                    <h3 style='color: #e2e8f0; margin-bottom: 1.5rem;'>Login</h3>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        login_btn = st.button("Login", use_container_width=True)
        
        if login_btn:
            is_authenticated, role = authenticate(username, password)
            if is_authenticated:
                st.session_state.authenticated = True
                st.session_state.user_role = role
                st.session_state.username = username
                st.session_state.user_logger = UserLogger(username)
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with col2:
        st.markdown("""
            <div style='padding: 2rem; color: #e2e8f0;'>
                <h2 style='color: #e2e8f0; margin-bottom: 1.5rem;'>About Virtualization</h2>
                <p style='margin-bottom: 1rem;'>
                    Virtualization is a fundamental technology that allows the creation of virtual versions of computing resources,
                    enabling multiple operating systems and applications to run on a single physical machine.
                </p>
                <h3 style='color: #e2e8f0; margin: 1rem 0;'>Key Benefits:</h3>
                <ul style='margin-bottom: 1.5rem;'>
                    <li>Resource Optimization</li>
                    <li>Enhanced Security</li>
                    <li>Improved Disaster Recovery</li>
                    <li>Cost Efficiency</li>
                    <li>Environmental Sustainability</li>
                </ul>
                <h3 style='color: #e2e8f0; margin: 1rem 0;'>Types of Virtualization:</h3>
                <ul style='margin-bottom: 1.5rem;'>
                    <li>Server Virtualization</li>
                    <li>Desktop Virtualization</li>
                    <li>Network Virtualization</li>
                    <li>Storage Virtualization</li>
                </ul>
                <p>
                    Our chat assistant helps you understand virtualization concepts, implementation strategies,
                    and best practices in both Windows and Linux environments.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.stop()

# Main application (only shown to authenticated users)
st.markdown(f"""
    <h1 style='margin-bottom: 0.5rem;'>Class Chat Assistant</h1>
    <p style='color: #64748B; margin-bottom: 2rem;'>Interactive Learning Assistant</p>
""", unsafe_allow_html=True)

# Sidebar for database and model selection
with st.sidebar:
    # User info section
    with st.container():
        st.markdown(f"""
            <div style='padding: 1rem; background-color: white; border-radius: 8px; border: 1px solid #E2E8F0; margin-bottom: 2rem;'>
                <h3 style='margin: 0; color: #1E3A8A;'>👤 {st.session_state.username}</h3>
                <p style='margin: 0; color: #64748B;'>{st.session_state.user_role.capitalize()} Account</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Settings")
    
    # View History in a modal
    if st.button("📚 View History", use_container_width=True):
        if st.session_state.user_logger:
            history = st.session_state.user_logger.get_user_history()
            with st.expander("Interaction History", expanded=True):
                st.text_area("", value=history, height=400, disabled=True)
    
    st.markdown("---")
    
    # Database operations mode - only show create mode to admin
    if st.session_state.user_role == "admin":
        operation_mode = st.radio(
            "Operation Mode",
            ["Create New Database", "Query Existing Database"],
            index=1,
            format_func=lambda x: "📝 " + x if x == "Create New Database" else "🔍 " + x
        )
    else:
        operation_mode = "Query Existing Database"
    
    if operation_mode == "Create New Database":
        with st.form("create_db_form"):
            st.markdown("### Create New Database")
            new_db_name = st.text_input(
                "Database Name",
                placeholder="e.g., lecture_7"
            )
            
            uploaded_file = st.file_uploader(
                "Upload Document",
                type=["txt", "pdf", "doc", "docx"]
            )
            
            text_input = st.text_area(
                "Or paste text directly",
                height=150,
                placeholder="Enter your text here..."
            )
            
            submit_button = st.form_submit_button("Create Database", use_container_width=True)
            
            if submit_button:
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
                        tmp_path.unlink()
                    else:
                        text_content = text_input
                    
                    vectordb = get_vectordb(new_db_name, embedding_manager.embeddings)
                    vectordb.store_documents([text_content])
                    st.session_state.vectordb = vectordb
                
                st.success(f"Database '{new_db_name}' created successfully!")
    
    else:  # Query Existing Database
        available_dbs = get_available_databases()
        
        if not available_dbs:
            st.warning("No existing databases found. Please create a new database first.")
            st.stop()
        
        with st.form("query_db_form"):
            st.markdown("### Query Database")
            selected_db = st.selectbox(
                "Select Database",
                options=available_dbs,
                format_func=lambda x: f"📚 {x}"
            )
            
            selected_model = st.selectbox(
                "Select Model",
                options=QUERY_MODELS,
                format_func=lambda x: f"🤖 {x}"
            )
            
            load_button = st.form_submit_button("Load Database", use_container_width=True)
            
            if load_button:
                with st.spinner("Loading database..."):
                    vectordb = get_vectordb(selected_db, embedding_manager.embeddings)
                    st.session_state.vectordb = vectordb
                    st.session_state.chain = llm_manager.create_chain(
                        vectordb,
                        model_name=selected_model
                    )
                    st.session_state.messages = []
                    st.session_state.source_documents = []
                st.success(f"Database '{selected_db}' loaded successfully!")
    
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_role = None
        st.session_state.username = None
        st.session_state.user_logger = None
        st.rerun()

# Main chat interface
st.divider()

# Create a container for the entire chat interface
chat_interface = st.container()

with chat_interface:
    # Create a container for chat messages
    message_container = st.container()
    
    # Query input at the bottom
    query = st.chat_input("Ask your question...", key="query_input")
    
    # Display messages in the container
    with message_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])
    
    if query and not st.session_state.processing:
        st.session_state.processing = True
        
        try:
            if st.session_state.vectordb is None:
                st.error("Please load a database first!")
                st.stop()
            
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Process query asynchronously
            with st.spinner("Processing your question..."):
                result = asyncio.run(process_query_async(query))
                st.session_state.current_result = result
                
                if result:
                    # Add assistant's response to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"]
                    })
                    if st.session_state.user_logger:
                        st.session_state.user_logger.log_interaction(
                            query=query,
                            response=result,
                            database_name=selected_db
                        )
            
            # Rerun to update the chat display
            st.rerun()
        
        finally:
            st.session_state.processing = False

# Display extracted content in an expander below the chat
if st.session_state.current_result and st.session_state.current_result.get("extracted_content"):
    with st.expander("📑 Source Information", expanded=False):
        st.markdown(f"*Found {st.session_state.current_result.get('source_count', 0)} relevant sources*")
        for i, content in enumerate(st.session_state.current_result["extracted_content"], 1):
            with st.container():
                st.markdown(f"**Source {i}:**")
                st.markdown(content["content"])
                if "relevance" in content:
                    st.markdown(f"*Relevance: {content['relevance']}*")
                if i < len(st.session_state.current_result["extracted_content"]):
                    st.divider() 