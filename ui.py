import asyncio
import logging
import os
import sys
import time

import pandas as pd
import streamlit as st
import tiktoken
from dotenv import load_dotenv
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.global_search.search import GlobalSearch

from global_context import GlobalCommunityContext
from vector_store import (load_data, setup_vector_store, get_embedder)


def setup_logging(
        console_level=logging.DEBUG,
        file_level=logging.DEBUG,
        log_file='app.log'
):
    """
    Set up logging configuration with both console and file handlers.

    Args:
        console_level: Logging level for console output
        file_level: Logging level for file output
        log_file: Path to log file
    """
    # Create a custom formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # Capture all levels

    # Clear any existing handlers
    root_logger.handlers = []

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Test logging
    root_logger.debug('Logging configured successfully')


# Get authentication credentials from environment variables
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")  # Default fallback for development
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")  # Default fallback for development


# Authentication function
def authenticate(username, password):
    """Verify username and password against environment variables."""
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD


# Initialize session state for authentication
def init_session_state():
    """Initialize session state variables."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = 0


# Login page
def login_page():
    """Display and handle login page."""
    st.title("Login to GraphRAG Chatbot")

    # Check if too many failed attempts
    if st.session_state.login_attempts >= 3:
        st.error("Too many failed login attempts. Please try again later.")
        time.sleep(5)  # Add delay to prevent brute force
        st.session_state.login_attempts = 0
        return

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.session_state.login_attempts = 0
                st.success("Login successful!")
                st.rerun()
            else:
                st.session_state.login_attempts += 1
                remaining_attempts = 3 - st.session_state.login_attempts
                st.error(f"Invalid username or password. {remaining_attempts} attempts remaining.")


# Load base documents data
@st.cache_data(show_spinner=False)
def load_base_documents(base_documents):
    """Load base documents from parquet file."""
    try:
        df = pd.read_parquet(base_documents)
        return df
    except Exception as e:
        st.error(f"Error loading base documents: {str(e)}")
        return None


def render_chat_interface():
    """Render the chat interface page."""
    # Chat interface sidebar options
    search_mode = "global-dynamic"

    input_dir = "indexing/output"

    response_type = "Single Paragraph"

    community_level = 4

    dynamic_config = {}

    dynamic_config['use_summary'] = True

    dynamic_config['vector_store_threshold'] = 0.5

    dynamic_config['threshold'] = 1
    dynamic_config['keep_parent'] = False

    dynamic_config['num_repeats'] = 1

    dynamic_config['max_level'] = 4

    dynamic_config['concurrent_coroutines'] = 20

    drift_config = {}

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    llm = ChatOpenAI(
        api_key=openai_api_key,
        model="gpt-4o-mini",
        api_type=OpenaiApiType.OpenAI,
        max_retries=20
    )

    token_encoder = tiktoken.get_encoding("cl100k_base")

    # Load data and setup vector store
    with st.spinner("Loading data and setting up vector store..."):
        entities, communities, reports, relationships, text_units = load_data(input_dir, community_level)
        description_embedding_store, entity_description_embeddings = setup_vector_store(
            input_dir,
            community_level
        )

    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle user input
    user_query = st.chat_input(placeholder="Ask me anything")

    if user_query:
        start_time = time.time()
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            search_engine = setup_dynamic_global_search(
                    llm, token_encoder, communities, reports, entities, response_type, dynamic_config
            )

            async def perform_search():
                result = await search_engine.asearch(user_query)
                return result

            with st.spinner("Searching for an answer..."):
                result = asyncio.run(perform_search())

            response = result.response
            st.session_state.messages.append({"role": "assistant", "content": response})
            if search_mode == "global" or search_mode == "global-dynamic":
                st.write(response)
                if 'reports' in result.context_data.keys():
                    with st.expander("View Source Data"):
                        st.write(result.context_data["reports"])

            # Display context data
            if 'sources' in result.context_data.keys():
                with st.expander("View Source Data"):
                    st.write(result.context_data['sources'])

            # Display LLM calls and tokens
            latency = round(result.completion_time, 2)

            st.write(f"LLM calls: {result.llm_calls}. LLM tokens: {result.prompt_tokens}, latency: {latency}s")

def setup_dynamic_global_search(llm, token_encoder, communities, reports, entities, response_type, dynamic_config):
    """Set up global search engine with dynamic community selection."""
    context_builder = GlobalCommunityContext(
        community_reports=reports,
        communities=communities,
        entities=entities,
        token_encoder=token_encoder,
        dynamic_community_selection=True,
        dynamic_community_selection_kwargs={
            "llm": llm,
            "token_encoder": token_encoder,
            "use_summary": dynamic_config['use_summary'],
            "threshold": dynamic_config['threshold'],
            "keep_parent": dynamic_config['keep_parent'],
            "num_repeats": dynamic_config['num_repeats'],
            "max_level": dynamic_config['max_level'],
            "concurrent_coroutines": dynamic_config['concurrent_coroutines'],
        }
    )

    context_builder_params = {
        "use_community_summary": False,
        "shuffle_data": True,
        "include_community_rank": True,
        "min_community_rank": 0,
        "community_rank_name": "rank",
        "include_community_weight": True,
        "community_weight_name": "occurrence weight",
        "normalize_community_weight": True,
        "max_tokens": 12_000,
        "context_name": "Reports",
    }

    map_llm_params = {
        "max_tokens": 1000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }

    reduce_llm_params = {
        "max_tokens": 2000,
        "temperature": 0.0,
    }

    return GlobalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,
        json_mode=False,
        context_builder_params=context_builder_params,
        concurrent_coroutines=10,
        response_type=response_type
    )

def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()

    # Check authentication
    if not st.session_state.authenticated:
        login_page()
        return

    st.title("Arabic Chatbot POC")



    try:
        render_chat_interface()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)  # This will show the full traceback in development


if __name__ == "__main__":
    load_dotenv()
    setup_logging()
    main()
