import arxiv
from pathlib import Path

def download_papers(topics, num_results_per_topic):
    """
    Download papers from arxiv for given topics and number of results per topic
    
    Args:
        topics (list): List of research topics
        num_results_per_topic (int): Number of papers to download per topic
    
    Returns:
        list: List of downloaded PDF file paths
    """
    client = arxiv.Client()
    downloaded_papers = []

    for topic in topics:
        search = arxiv.Search(
            query=topic,
            max_results=num_results_per_topic,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        results = client.results(search)

        for r in results:
            pdf_path = r.download_pdf()
            downloaded_papers.append(pdf_path)

    return downloaded_papers

def list_pdf_files(directory='.'):
    """
    List PDF files in a given directory
    
    Args:
        directory (str, optional): Directory path. Defaults to current directory.
    
    Returns:
        list: List of PDF file names
    """
    pdf_files = [file.name for file in Path(directory).glob('*.pdf')]
    return pdf_files