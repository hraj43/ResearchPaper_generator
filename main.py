import os
from dotenv import load_dotenv
import asyncio
import time

import nest_asyncio
from llama_index.llms.openai import OpenAI

from src.config import RESEARCH_PAPER_TOPICS, NUM_RESULTS_PER_TOPIC
from src.arxiv_downloader import download_papers, list_pdf_files
from src.document_parser import parse_pdf_files
from src.llama_cloud_pipeline import create_llamacloud_pipeline, upload_documents
from src.query_engine import create_query_engine
from src.report_generator import ReportGenerationAgent

# Load environment variables
load_dotenv()

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

async def generate_outline_from_query(query, llm):
    """
    Generate a research paper outline based on a user query.
    
    Args:
        query (str): The research topic or question.
        llm: Language model to use for outline generation.
        
    Returns:
        str: A structured outline for the research paper.
    """
    prompt = f"""
    Generate a detailed research paper outline for the following topic: "{query}"
    
    The outline should follow this format exactly:
    
    # Research Paper Report on [Topic]

    ## 1. Introduction

    ## 2. [Background/Fundamentals Section]
    2.1. [Subsection]
    2.2. [Subsection]

    ## 3. Latest Papers:
    3.1. [Specific Research Area or Paper Topic]
    3.2. [Specific Research Area or Paper Topic]
    3.3. [Specific Research Area or Paper Topic]

    ## 4. Conclusion
    
    Replace the bracketed text with relevant content for the query topic.
    Ensure the outline is comprehensive, logical, and focused on the query topic.
    """
    
    response = await llm.acomplete(prompt)
    outline = response.text.strip()
    
    # Ensure the outline has proper formatting
    if not outline.startswith("# Research Paper"):
        outline = f"# Research Paper Report on {query}\n\n" + outline
    
    # Make sure there's a conclusion section
    if "## Conclusion" not in outline and "## 4. Conclusion" not in outline:
        outline += "\n\n## 4. Conclusion"
    
    return outline

async def initialize_research_pipeline(query, model="gpt-3.5-turbo", max_retries=3):
    """
    Initialize the research pipeline and generate report based on a query.
    
    Args:
        query (str): Research topic query to generate an outline from.
        model (str, optional): OpenAI model to use. Defaults to "gpt-3.5-turbo".
        max_retries (int, optional): Maximum number of retries for generation. Defaults to 3.
    
    Returns:
        dict: Generated report
    """
    # Check for API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    
    if not openai_api_key or not llama_cloud_api_key:
        raise ValueError("Missing API keys. Please set OPENAI_API_KEY and LLAMA_CLOUD_API_KEY in .env file.")

    # Initialize language model with a valid model name
    llm = OpenAI(
        api_key=openai_api_key, 
        model=model,
        temperature=0.3,
        max_tokens=4000  # Ensure sufficient tokens for complete generation
    )
    
    # Generate outline from query
    print(f"Generating outline based on query: '{query}'")
    outline = await generate_outline_from_query(query, llm)
    print("Generated outline:")
    print(outline)
    print("\n")


    # Download research papers
    downloaded_papers = download_papers(RESEARCH_PAPER_TOPICS, NUM_RESULTS_PER_TOPIC)
    pdf_files = list_pdf_files()
    print(f"Found {len(pdf_files)} PDF files")

    # Parse documents
    print("Parsing PDF files...")
    documents = parse_pdf_files(pdf_files)
    print(f"Parsed {len(documents)} documents")

    # Embedding and transformation configurations
    embedding_config = {
        'type': 'OPENAI_EMBEDDING',
        'component': {
            'api_key': openai_api_key,
            'model_name': 'text-embedding-ada-002'
        }
    }

    transform_config = {
        'mode': 'auto',
        'config': {
            'chunk_size': 1024,
            'chunk_overlap': 20
        }
    }

    # Create LlamaCloud pipeline
    print("Creating LlamaCloud pipeline...")
    client, pipeline = create_llamacloud_pipeline('report_generation', embedding_config, transform_config)

    # Upload documents
    print("Uploading documents to LlamaCloud...")
    await upload_documents(client, pipeline, documents, llm)

    # Create query engine
    print("Creating query engine...")
    query_engine = create_query_engine(llama_cloud_api_key)

    # Process the outline into sections for potential section-by-section generation
    sections = parse_outline_sections(outline)
    
    # Initialize report generation agent
    agent = ReportGenerationAgent(
        query_engine=query_engine,
        llm=llm,
        verbose=True,
        timeout=2400.0  # Extend timeout to 40 minutes
    )
   
    # Attempt generation with retries
    for attempt in range(max_retries):
        try:
            print(f"Generating report (Attempt {attempt+1}/{max_retries})...")
            report = await agent.run(outline=outline)
            
            # Verify we have a complete report by checking for conclusion
            if 'response' in report and report['response'] and "Conclusion" in report['response']:
                return {
                    "response": report['response'],
                    "outline": outline,
                    "success": True
                }
            else:
                print("Warning: Generated report appears incomplete. Retrying...")
                # Short delay before retry
                time.sleep(5) 
        except Exception as e:
            print(f"Error in generation attempt {attempt+1}: {str(e)}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(10)  # Wait longer after an error
            else:
                raise
    
    # If we reach here without returning, try fallback to section-by-section generation
    print("Attempting fallback to section-by-section generation...")
    full_report = await generate_report_by_sections(agent, sections)
    return {
        "response": full_report,
        "outline": outline,
        "success": True
    }

def parse_outline_sections(outline):
    """Parse outline into sections for separate generation"""
    lines = outline.strip().split('\n')
    sections = []
    current_section = []
    
    for line in lines:
        if line.startswith('## '):
            if current_section:
                sections.append('\n'.join(current_section))
            current_section = [line]
        else:
            current_section.append(line)
    
    if current_section:
        sections.append('\n'.join(current_section))
    
    return sections

async def generate_report_by_sections(agent, sections):
    """Generate report section by section as a fallback approach"""
    full_report = []
    
    # Generate the title and intro first
    intro_outline = sections[0]
    if len(sections) > 1:
        intro_outline += "\n" + sections[1]
    
    intro_result = await agent.run(outline=intro_outline)
    if 'response' in intro_result and intro_result['response']:
        full_report.append(intro_result['response'])
    
    # Generate each content section
    for i in range(2, len(sections)):
        section_outline = sections[i]
        try:
            section_result = await agent.run(outline=section_outline)
            if 'response' in section_result and section_result['response']:
                # Extract just the section content without repeating headers
                section_content = extract_section_content(section_result['response'])
                full_report.append(section_content)
        except Exception as e:
            print(f"Error generating section {i}: {str(e)}")
            # Add placeholder if section generation fails
            full_report.append(f"\n## {section_outline.strip().split()[1]}\n\nContent generation incomplete for this section.\n")
    
    return "\n\n".join(full_report)

def extract_section_content(section_text):
    """Extract section content without title and intro duplication"""
    lines = section_text.strip().split('\n')
    start_idx = 0
    
    # Find where actual section content starts, skipping any title and intro
    for i, line in enumerate(lines):
        if line.startswith('## '):
            start_idx = i
            break
    
    return '\n'.join(lines[start_idx:])

# CLI version for testing
if __name__ == "__main__":
    try:
        # Default model 
        model_to_use = "gpt-3.5-turbo"
        
        # Get query from user input
        user_query = input("Enter your research topic query (e.g., 'quantum computing applications in cybersecurity'): ")
        
        print(f"Generating research paper on: {user_query}")
        print(f"Using model: {model_to_use}")
        
        report = asyncio.run(initialize_research_pipeline(query=user_query, model=model_to_use))
        
        if report and 'response' in report and report['response']:
            print("\nReport generated successfully!")
            
            # Check report completeness
            content = report['response']
            sections_found = content.count('##')
            print(f"Found {sections_found} major sections in the report")
            
            # Save to file
            with open("research_paper.md", "w", encoding="utf-8") as f:
                f.write(report['response'])
            print("Report saved to research_paper.md")
            
            # Preview the report structure
            print("\nReport Structure Preview:")
            lines = content.split('\n')
            for line in lines:
                if line.startswith('#'):
                    print(line)
        else:
            print("Error: Generated report is empty or invalid")
    except Exception as e:
        print(f"An error occurred: {str(e)}")