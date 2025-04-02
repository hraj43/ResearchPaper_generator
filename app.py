import streamlit as st
import asyncio
import sys
import os
from io import StringIO
import time
import threading
import tempfile
from src.config import OPENAI_API_KEY,LLAMA_CLOUD_API_KEY

# Import the research generator module
from main import initialize_research_pipeline

# Set page configuration
st.set_page_config(
    page_title="AI Research Paper Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Function to run async code in a thread and capture output
def run_async_in_thread(coro):
    # Create a StringIO object to capture prints
    captured_output = StringIO()
    original_stdout = sys.stdout
    sys.stdout = captured_output
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(coro)
    loop.close()
    
    # Restore stdout
    sys.stdout = original_stdout
    return result, captured_output.getvalue()

# Function to display progress and logs
def display_logs_and_progress(progress_bar, status_text, log_container, stop_event):
    progress_value = 0
    stages = [
        "Generating outline...",
        "Downloading research papers...",
        "Parsing documents...",
        "Creating pipeline...",
        "Uploading documents...",
        "Creating query engine...",
        "Generating report..."
    ]
    
    stage_index = 0
    while not stop_event.is_set() and stage_index < len(stages):
        # Update status text
        status_text.text(stages[stage_index])
        
        # Update progress based on current stage
        progress_step = 1.0 / len(stages)
        target_progress = (stage_index + 1) * progress_step
        
        # Smooth progress animation
        while progress_value < target_progress and not stop_event.is_set():
            progress_value = min(progress_value + 0.01, target_progress)
            progress_bar.progress(progress_value)
            time.sleep(0.1)
        
        stage_index += 1
        
        # Add some randomness to make it feel natural
        if stage_index < len(stages):
            time.sleep(2 + stage_index * 0.5)
    
    # Ensure we reach 100% when done
    if not stop_event.is_set():
        progress_bar.progress(1.0)
        status_text.text("Research paper generated successfully!")

def main():
    # Manually set API keys
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["LLAMA_CLOUD_API_KEY"] =LLAMA_CLOUD_API_KEY
    
    # Title and introduction
    st.markdown('<div class="main-header">🔬 AI Research Paper Generator</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-text">
    This application generates research papers based on previous studies related to your query. By leveraging a RAG-based system, it retrieves and analyzes existing research to provide valuable insights, helping you understand past work and develop your own research more effectively.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configurations
    st.sidebar.header("Settings")
    
    # Model selection
    model = st.sidebar.selectbox(
        "Select Language Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        index=0
    )
    
    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        max_retries = st.slider(
            "Maximum Generation Attempts",
            min_value=1,
            max_value=5,
            value=3,
            help="Number of attempts the system will make to generate a complete paper"
        )
    
    # Main content area
    query = st.text_area(
        "Enter your research topic:", 
        height=100,
        placeholder="e.g., quantum computing applications in healthcare, or recent advances in natural language processing"
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        generate_button = st.button("Generate Research Paper", type="primary", use_container_width=True)
    
    # Initialize session state for storing results
    if 'paper_content' not in st.session_state:
        st.session_state.paper_content = None
    if 'outline_content' not in st.session_state:
        st.session_state.outline_content = None
    
    if generate_button and query:
        # Create containers for output
        progress_container = st.container()
        log_container = st.container()
        output_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
        
        with log_container:
            st.markdown('<div class="subheader">Generation Process</div>', unsafe_allow_html=True)
            log_output = st.empty()
        
        # Start a thread for the progress animation
        stop_event = threading.Event()
        progress_thread = threading.Thread(
            target=display_logs_and_progress,
            args=(progress_bar, status_text, log_output, stop_event)
        )
        progress_thread.start()
        
        try:
            # Run the paper generation in a separate thread
            with st.spinner("Generating research paper..."):
                # Execute the paper generation
                result, logs = run_async_in_thread(
                    initialize_research_pipeline(
                        query=query, 
                        model=model, 
                        max_retries=max_retries
                    )
                )
            
            # Update log output
            log_output.code(logs, language="")
            
            # Signal the progress thread to stop
            stop_event.set()
            progress_thread.join()
            
            # Display the final paper
            if result and 'response' in result and result['response']:
                # Store in session state
                st.session_state.paper_content = result['response']
                st.session_state.outline_content = result.get('outline', '')
                
                with output_container:
                    st.markdown('<div class="subheader">Generated Research Paper</div>', unsafe_allow_html=True)
                    
                    # Show tabs for outline and full paper
                    tab1, tab2 = st.tabs(["Full Paper", "Outline"])
                    
                    with tab1:
                        st.markdown(result['response'])
                        
                        # Download button for the paper
                        st.download_button(
                            label="Download Paper (Markdown)",
                            data=result['response'],
                            file_name="research_paper.md",
                            mime="text/markdown"
                        )
                    
                    with tab2:
                        if 'outline' in result:
                            st.markdown(result['outline'])
                        else:
                            st.info("Outline information not available")
            else:
                st.error("Failed to generate a complete paper. Please try again or modify your query.")
        
        except Exception as e:
            # Signal the progress thread to stop on error
            stop_event.set()
            progress_thread.join()
            
            st.error(f"An error occurred: {str(e)}")
            log_output.code(logs, language="")
    
    # Show previously generated content if available
    elif st.session_state.paper_content:
        output_container = st.container()
        with output_container:
            st.markdown('<div class="subheader">Previously Generated Research Paper</div>', unsafe_allow_html=True)
            
            # Show tabs for outline and full paper
            tab1, tab2 = st.tabs(["Full Paper", "Outline"])
            
            with tab1:
                st.markdown(st.session_state.paper_content)
                
                # Download button for the paper
                st.download_button(
                    label="Download Paper (Markdown)",
                    data=st.session_state.paper_content,
                    file_name="research_paper.md",
                    mime="text/markdown"
                )
            
            with tab2:
                if st.session_state.outline_content:
                    st.markdown(st.session_state.outline_content)
                else:
                    st.info("Outline information not available")

if __name__ == "__main__":
    main()