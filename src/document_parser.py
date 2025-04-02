from llama_parse import LlamaParse

def parse_pdf_files(pdf_files, result_type="markdown", num_workers=4):
    """
    Parse PDF files using LlamaParse
    
    Args:
        pdf_files (list): List of PDF file paths
        result_type (str, optional): Parse result type. Defaults to "markdown".
        num_workers (int, optional): Number of workers for parsing. Defaults to 4.
    
    Returns:
        list: Parsed documents
    """
    parser = LlamaParse(
        result_type=result_type,
        num_workers=num_workers,
        verbose=True,
    )

    documents = []

    for index, pdf_file in enumerate(pdf_files):
        print(f"Processing file {index + 1}/{len(pdf_files)}: {pdf_file}")
        document = parser.load_data(pdf_file)
        documents.append(document)

    return documents