import streamlit as st
import requests
import xml.etree.ElementTree as ET
import asyncio
import sys
import os
import io
from io import StringIO
import time
import threading
import tempfile
from src.config import OPENAI_API_KEY, LLAMA_CLOUD_API_KEY

# Import the research generator module
from main import initialize_research_pipeline, list_pdf_files, generate_outline_from_query

def fetch_arxiv_papers(query, max_results=3):
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        root = ET.fromstring(response.text)
        papers = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text
            pdf_link = entry.find("{http://www.w3.org/2005/Atom}link[@title='pdf']").attrib["href"]
            papers.append({"title": title, "pdf_link": pdf_link})
        return papers
    return []

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
def display_logs_and_progress(progress_bar, status_text, log_container, stop_event, stages):
    progress_value = 0
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
            time.sleep(1 + stage_index * 0.3)
    
    # Ensure we reach 100% when done
    if not stop_event.is_set():
        progress_bar.progress(1.0)
        status_text.text("Process completed successfully!")

# Function to parse outline into sections and subsections
def parse_outline(outline_text):
    sections = []
    current_section = {}
    current_subsections = []
    
    for line in outline_text.strip().split('\n'):
        if line.startswith('# '):
            # Title line, just skip
            continue
        elif line.startswith('## '):
            # If we have a previous section, save it
            if current_section:
                current_section['subsections'] = current_subsections
                sections.append(current_section)
                
            # Start a new section
            current_section = {'title': line.strip('## ').strip(), 'line': line, 'key': f"section_{len(sections)}"}
            current_subsections = []
        elif line.startswith('### ') or any(line.startswith(f'{i}.') for i in range(1, 10)):
            # Subsection
            current_subsections.append({'title': line.strip('### ').strip('0123456789. ').strip(), 'line': line, 'key': f"subsection_{len(sections)}_{len(current_subsections)}"})
    
    # Add the last section if exists
    if current_section:
        current_section['subsections'] = current_subsections
        sections.append(current_section)
        
    return sections

# Set page configuration
st.set_page_config(
    page_title="Research Paper Content Generation",
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
    .section-selector {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Manually set API keys
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["LLAMA_CLOUD_API_KEY"] = LLAMA_CLOUD_API_KEY
    
    # Title and introduction
    st.markdown('<div class="main-header">🔬 Research Content Generator</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-text">
    This application generates research papers based on previous studies related to your query. By leveraging a RAG-based system, it retrieves and analyzes existing research to provide valuable insights, helping you understand past work and develop your own research more effectively.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state if not present
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "uploaded_pdfs_store" not in st.session_state:
        st.session_state.uploaded_pdfs_store = {}  # Store uploaded PDFs separately
    if "fetched_pdfs_store" not in st.session_state:
        st.session_state.fetched_pdfs_store = {}  # Store fetched PDFs separately
    if "stored_pdfs" not in st.session_state:
        st.session_state.stored_pdfs = []  # Store fetched research papers separately
    if "outline" not in st.session_state:
        st.session_state.outline = None
    if "parsed_outline" not in st.session_state:
        st.session_state.parsed_outline = []
    if "selected_sections" not in st.session_state:
        st.session_state.selected_sections = {}
    if "paper_content" not in st.session_state:
        st.session_state.paper_content = None
    if "current_page" not in st.session_state:
        st.session_state.current_page = "main"  # Can be "main", "outline", or "content"

    # Sidebar: Upload PDFs
    with st.sidebar:
        st.subheader("📂 Upload Your Documents")
        uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)

    # Detect removed files and update session state **only for uploaded PDFs**
    current_uploaded_names = {pdf.name for pdf in uploaded_files} if uploaded_files else set()
    stored_uploaded_names = set(st.session_state.uploaded_pdfs_store.keys())

    for stored_pdf in stored_uploaded_names - current_uploaded_names:
        del st.session_state.uploaded_pdfs_store[stored_pdf]

    # Store newly uploaded PDFs in session state
    if uploaded_files:
        for pdf in uploaded_files:
            if pdf.name not in st.session_state.uploaded_pdfs_store:
                st.session_state.uploaded_pdfs_store[pdf.name] = io.BytesIO(pdf.read())

    # Sidebar: Fetch PDFs from ArXiv
    with st.sidebar:
        st.subheader("📄 Fetch Research Papers")
        
        query = st.text_input("🔍 Enter search query")
        if query:
            st.session_state.query = query

        if st.button("Fetch Papers"):
            # Fetch papers from ArXiv
            st.session_state.stored_pdfs = fetch_arxiv_papers(query)

            # Download PDFs & store in memory (separately from uploads)
            for paper in st.session_state.stored_pdfs:
                response = requests.get(paper["pdf_link"])
                if response.status_code == 200:
                    st.session_state.fetched_pdfs_store[paper["title"]] = io.BytesIO(response.content)

    # MAIN PAGE LAYOUT
    if st.session_state.current_page == "main":
        # Display fetched PDFs in a clean format
        if st.session_state.stored_pdfs:
            st.subheader("Fetched Research Papers")

            for paper in list(st.session_state.stored_pdfs):  # Convert to list to avoid modification issues
                col1, col2, col3 = st.columns([0.8, 0.1, 0.1])

                with col1:
                    st.markdown(f"📄 {paper['title'][:40]}...", unsafe_allow_html=True)

                with col2:
                    st.markdown(
                        f'<a href="{paper["pdf_link"]}" target="_blank" style="text-decoration: none; color: black; font-size: 18px;">👁️</a>',
                        unsafe_allow_html=True
                    )

                with col3:
                    if st.button("✖", key=f"remove_{paper['pdf_link']}"):
                        st.session_state.stored_pdfs = [
                            p for p in st.session_state.stored_pdfs if p["pdf_link"] != paper["pdf_link"]
                        ]
                        
                        # Remove from fetched PDFs store safely
                        if paper["title"] in st.session_state.fetched_pdfs_store:
                            del st.session_state.fetched_pdfs_store[paper["title"]]

                        st.rerun()

        # Merge uploaded and fetched PDFs into a single dictionary for parsing
        st.session_state.pdf_data_store = {
            **st.session_state.uploaded_pdfs_store,  # Include uploaded PDFs
            **st.session_state.fetched_pdfs_store   # Include fetched PDFs
        }

        # ✅ Concatenate all PDFs into a single variable
        all_pdfs = list(st.session_state.pdf_data_store.values())  # List of `io.BytesIO` PDF objects

        st.write(f"⬅️ **Upload or fetch your Papers**")
        st.write(f"Total PDFs stored: {len(all_pdfs)}")

        col1, col2 = st.columns([1, 3])
        with col1:
            generate_outline_button = st.button("Generate Outline", type="primary", use_container_width=True)
        
        if generate_outline_button and st.session_state.query and len(all_pdfs) > 0:
            # Create containers for output
            progress_container = st.container()
            log_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0.0)
                status_text = st.empty()
            
            with log_container:
                st.markdown('<div class="subheader">Generation Process</div>', unsafe_allow_html=True)
                log_output = st.empty()
            
            # Define stages for outline generation
            outline_stages = [
                "Analyzing documents...",
                "Generating outline structure...",
                "Finalizing outline..."
            ]
            
            # Start a thread for the progress animation
            stop_event = threading.Event()
            progress_thread = threading.Thread(
                target=display_logs_and_progress,
                args=(progress_bar, status_text, log_output, stop_event, outline_stages)
            )
            progress_thread.start()
            
            try:
                # Run the outline generation in a separate thread
                with st.spinner("Generating Outline..."):
                    # Get the OpenAI client
                    from llama_index.llms.openai import OpenAI
                    llm = OpenAI(
                        api_key=OPENAI_API_KEY, 
                        model="gpt-3.5-turbo",
                        temperature=0.3
                    )
                    
                    # Execute outline generation
                    outline, logs = run_async_in_thread(
                        generate_outline_from_query(
                            query=st.session_state.query, 
                            llm=llm
                        )
                    )
                
                # Signal the progress thread to stop
                stop_event.set()
                progress_thread.join()
                
                # Store the outline in session state
                st.session_state.outline = outline
                st.session_state.parsed_outline = parse_outline(outline)
                
                # Initialize all sections as selected by default
                st.session_state.selected_sections = {}
                for i, section in enumerate(st.session_state.parsed_outline):
                    st.session_state.selected_sections[section['key']] = True
                    for j, subsection in enumerate(section['subsections']):
                        st.session_state.selected_sections[subsection['key']] = True
                
                # Move to outline page
                st.session_state.current_page = "outline"
                st.rerun()
                
            except Exception as e:
                # Signal the progress thread to stop on error
                stop_event.set()
                progress_thread.join()
                
                st.error(f"An error occurred: {str(e)}")
                log_output.code(logs if 'logs' in locals() else "", language="")

    # OUTLINE PAGE
    elif st.session_state.current_page == "outline":
        st.subheader("Research Paper Outline")
        st.info("Select the sections and subsections you want to include in your paper, then click 'Generate Selected Content'")
        
        # Back button
        if st.button("← Back to Main Page"):
            st.session_state.current_page = "main"
            st.rerun()
        
        # Display the full outline first for reference
        with st.expander("View Full Outline", expanded=False):
            st.markdown(st.session_state.outline)
        
        st.subheader("Select Sections")
        
        # Display sections with checkboxes
        for i, section in enumerate(st.session_state.parsed_outline):
            st.markdown(f"""<div class="section-selector">""", unsafe_allow_html=True)
            
            # Section checkbox
            section_selected = st.checkbox(
                f"## {section['title']}", 
                value=st.session_state.selected_sections.get(section['key'], True),
                key=section['key']
            )
            st.session_state.selected_sections[section['key']] = section_selected
            
            # Subsections (indented)
            if section_selected and section['subsections']:
                cols = st.columns([0.1, 0.9])
                with cols[1]:
                    for subsection in section['subsections']:
                        subsection_selected = st.checkbox(
                            subsection['line'], 
                            value=st.session_state.selected_sections.get(subsection['key'], True),
                            key=subsection['key']
                        )
                        st.session_state.selected_sections[subsection['key']] = subsection_selected
            
            st.markdown("""</div>""", unsafe_allow_html=True)
        
        # Generate content button
        generate_content_col1, generate_content_col2 = st.columns([1, 3])
        with generate_content_col1:
            generate_content_button = st.button("Generate Selected Content", type="primary", use_container_width=True)
        
        if generate_content_button:
            # First, filter the outline to only include selected sections
            filtered_outline = "# Research Paper Report\n\n"
            
            for section in st.session_state.parsed_outline:
                if st.session_state.selected_sections.get(section['key'], False):
                    filtered_outline += f"## {section['title']}\n\n"
                    
                    # Add selected subsections
                    for subsection in section['subsections']:
                        if st.session_state.selected_sections.get(subsection['key'], False):
                            filtered_outline += f"{subsection['line']}\n\n"
            
            # Create containers for output
            progress_container = st.container()
            log_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0.0)
                status_text = st.empty()
            
            with log_container:
                st.markdown('<div class="subheader">Generation Process</div>', unsafe_allow_html=True)
                log_output = st.empty()
            
            # Define stages for content generation
            content_stages = [
                "Generating outline...",
                "Downloading research papers...",
                "Parsing documents...",
                "Creating pipeline...",
                "Uploading documents...",
                "Creating query engine...",
                "Generating report..."
            ]
            
            # Start a thread for the progress animation
            stop_event = threading.Event()
            progress_thread = threading.Thread(
                target=display_logs_and_progress,
                args=(progress_bar, status_text, log_output, stop_event, content_stages)
            )
            progress_thread.start()
            
            try:
                # Get all PDFs
                all_pdfs = list(st.session_state.pdf_data_store.values())
                
                # Run the paper generation in a separate thread
                with st.spinner("Generating Paper Content..."):
                    # Execute the paper generation with the filtered outline
                    result, logs = run_async_in_thread(
                        initialize_research_pipeline(
                            query=st.session_state.query, 
                            pdf=all_pdfs,
                            model="gpt-3.5-turbo", 
                            max_retries=1,
                            custom_outline=filtered_outline  # Pass the filtered outline
                        )
                    )
                
                # Update log output
                log_output.code(logs, language="")
                
                # Signal the progress thread to stop
                stop_event.set()
                progress_thread.join()
                
                # Store the generated content
                if result and 'response' in result and result['response']:
                    st.session_state.paper_content = result['response']
                    
                    # Move to content page
                    st.session_state.current_page = "content"
                    st.rerun()
                else:
                    st.error("Failed to generate content. Please try again.")
                    
            except Exception as e:
                # Signal the progress thread to stop on error
                stop_event.set()
                progress_thread.join()
                
                st.error(f"An error occurred: {str(e)}")
                log_output.code(logs if 'logs' in locals() else "", language="")

    # CONTENT PAGE
    elif st.session_state.current_page == "content":
        st.subheader("Generated Research Paper")
        
        # Back button
        if st.button("← Back to Outline"):
            st.session_state.current_page = "outline"
            st.rerun()
        
        # Display the generated content
        st.markdown(st.session_state.paper_content)
        
        # Download button for the paper
        st.download_button(
            label="Download Paper (Markdown)",
            data=st.session_state.paper_content,
            file_name="research_paper.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()