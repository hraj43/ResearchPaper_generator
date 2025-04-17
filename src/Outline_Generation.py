import io
import re
from typing import List, Dict, Any, Tuple
from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.llms.openai import OpenAI

def hierarchical_chunk_pdf(pdf_data: io.BytesIO) -> List[Document]:
    """
    Process a PDF using hierarchical chunking to preserve document structure.
    
    Args:
        pdf_data: BytesIO object containing PDF data
        
    Returns:
        List of Document objects with hierarchical structure
    """
    try:
        from PyPDF2 import PdfReader
        
        # Extract text from PDF
        pdf_reader = PdfReader(pdf_data)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        # Create hierarchical node parser
        # This will chunk text while respecting document hierarchy
        hierarchical_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128],  # Size of chunks at each level
            chunk_overlap=20               # Overlap between chunks
            # Removed paragraph_separator parameter which caused the error
        )
        
        # Create document
        doc = Document(text=text)
        
        # Parse into hierarchical nodes
        nodes = hierarchical_parser.get_nodes_from_documents([doc])
        
        return nodes
    except Exception as e:
        print(f"Error in hierarchical chunking: {str(e)}")
        return []

def extract_outline_from_nodes(nodes, query: str = None, openai_api_key: str = None) -> Tuple[str, str]:
    """
    Extract outline from document nodes in standardized academic format.
    
    Args:
        nodes: List of document nodes
        query: Optional query to focus the outline
        openai_api_key: API key for OpenAI
        
    Returns:
        Tuple of (outline, logs)
    """
    logs = "Analyzing document structure...\n"
    
    # Extract content samples from nodes for LLM analysis
    content_samples = []
    for i, node in enumerate(nodes):
        if i >= 30:  # Limit to first 30 nodes to avoid token limits
            break
            
        # Ensure we have a string representation of the text
        if isinstance(node.text, tuple):
            text = str(node.text[0]) if node.text else ""
        else:
            text = str(node.text)
        
        text = text.strip()
        if text:
            # Get a sample of the content (first 200 chars)
            content_samples.append(text[:200] + "..." if len(text) > 200 else text)
    
    logs += f"Extracted content samples from {len(content_samples)} nodes\n"
    
    # Use LLM to structure the outline
    llm = OpenAI(
        api_key=openai_api_key, 
        model="gpt-3.5-turbo",
        temperature=0.3
    )
    
    # Format content samples for LLM processing
    content_text = "\n\n".join([f"Sample {i+1}:\n{sample}" for i, sample in enumerate(content_samples[:10])])
    
    # Standard academic outline structure template
    standard_outline_template = """
* **Abstract**
   * Brief summary of the research including objectives, methodology, and key results.
* **Introduction**
   * **Problem Statement**: What problem is being addressed?
   * **Objective of the Study**: Main goal of the research.
   * **Significance**: Why is this study important?
   * **Scope**: What is included and excluded from this study?
   * **Overview of the Paper**: A brief description of each subsequent section.
* **Literature Review**
   * **Existing Work**: Overview of previous research related to the topic.
   * **Theoretical Framework**: The theories or models guiding the research.
   * **Gap in the Literature**: What gaps are identified and how does the research fill them?
* **Methodology**
   * **Research Design**: Type of research (e.g., experimental, qualitative).
   * **Data Collection**: How data was gathered (e.g., surveys, experiments).
   * **Data Analysis**: Techniques used to analyze the data (e.g., statistical analysis, coding).
   * **Limitations**: Potential limitations in the study methodology.
* **Results**
   * **Presentation of Findings**: What was discovered during the research.
   * **Statistical Analysis**: If applicable, statistical results (e.g., p-values, correlation coefficients).
   * **Tables and Figures**: Relevant data visualizations.
* **Discussion**
   * **Interpretation of Results**: What do the results mean?
   * **Implications**: How do the results contribute to the field or solve the problem?
   * **Comparison with Previous Work**: How does the research align or differ from earlier studies?
* **Conclusion**
   * **Summary of Key Findings**: Recap of the main results.
   * **Contributions to Knowledge**: What new insights or contributions does the study offer?
   * **Future Research**: What areas could future studies explore?
* **References**
   * **Bibliography**: Citations of all sources referenced throughout the paper.
* **Appendices** (if applicable)
   * **Additional Materials**: Any supplementary data or materials that support the research.
"""
    
    # Create LLM prompt
    prompt = f"""
    I'm analyzing a research paper and need to generate an outline following a standard academic format.
    
    Here are some content samples from the paper:
    
    {content_text}
    
    Please create an outline following this EXACT structure, but adapt it to reflect the content of the specific paper:
    
    {standard_outline_template}
    
    Add relevant details from the paper to each section. If certain sections don't seem to exist in the paper, still include them in the outline but note they may need more development.
    
    If a specific research query was provided, focus the outline details around: {query if query else 'the general topic of the paper'}
    
    Return only the outline in markdown format, using the exact same structure and headings as the template.
    """
    
    logs += "Generating standardized academic outline...\n"
    
    # Handle the LLM response safely
    response = llm.complete(prompt)
    
    # Handle different response formats
    if hasattr(response, 'text'):
        outline = response.text
    elif isinstance(response, tuple):
        outline = str(response[0]) if response else ""
    elif isinstance(response, str):
        outline = response
    else:
        outline = str(response)
    
    logs += "Outline generation complete.\n"
    return outline, logs

def parse_outline(outline_text: str) -> Dict[str, Any]:
    """
    Parse outline text into structured format.
    
    Args:
        outline_text: Text of the outline
        
    Returns:
        Dictionary representing the outline structure
    """
    # Ensure outline_text is a string
    if not isinstance(outline_text, str):
        outline_text = str(outline_text)
    
    # Create structured representation
    outline_structure = {"sections": []}
    
    # Pattern to identify main sections (starts with * or - and has ** or __ around the title)
    main_section_pattern = re.compile(r'^\s*[\*\-]\s+[\*\_]{2}([^*_]+)[\*\_]{2}')
    
    # Pattern to identify subsections (starts with multiple spaces, then * or - and has ** or __ or nothing around the title)
    subsection_pattern = re.compile(r'^\s+[\*\-]\s+([\*\_]{2}([^*_]+)[\*\_]{2}|([^:]+))(?::)?(.+)?')
    
    current_section = None
    lines = outline_text.strip().split('\n')
    
    for line in lines:
        if not isinstance(line, str):
            line = str(line)
            
        # Check for main section
        main_match = main_section_pattern.match(line)
        if main_match:
            section_title = main_match.group(1).strip()
            current_section = {
                "title": section_title,
                "key": f"section_{len(outline_structure['sections'])}",
                "subsections": []
            }
            outline_structure["sections"].append(current_section)
            continue
            
        # Check for subsection
        sub_match = subsection_pattern.match(line)
        if sub_match and current_section is not None:
            # Try to extract both the subsection title and description
            if sub_match.group(2):  # Bold format with **title**
                subsection_title = sub_match.group(2).strip()
            elif sub_match.group(3):  # No bold format
                subsection_title = sub_match.group(3).strip()
            else:
                subsection_title = "Untitled subsection"
                
            description = sub_match.group(4).strip() if sub_match.group(4) else ""
            
            subsection = {
                "title": subsection_title,
                "key": f"subsection_{len(outline_structure['sections'])-1}_{len(current_section['subsections'])}",
                "description": description
            }
            current_section["subsections"].append(subsection)
    
    return outline_structure

async def generate_outline_from_pdf(pdf_data: io.BytesIO, query: str = None, openai_api_key: str = None):
    """
    Generate an outline from a PDF document using hierarchical chunking.
    
    Args:
        pdf_data: BytesIO object containing PDF data
        query: Optional query to focus the outline
        openai_api_key: API key for OpenAI
        
    Returns:
        Tuple of (outline, logs)
    """
    logs = "Starting hierarchical PDF analysis...\n"
    
    # Perform hierarchical chunking
    nodes = hierarchical_chunk_pdf(pdf_data)
    logs += f"Created {len(nodes)} hierarchical chunks from document\n"
    
    # Extract outline structure
    outline, outline_logs = extract_outline_from_nodes(nodes, query, openai_api_key)
    logs += outline_logs
    
    return outline, logs