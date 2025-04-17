from llama_parse import LlamaParse
import tempfile
import os

def parse_pdf_files(pdf_files, result_type="markdown", num_workers=4):
    """
    Parse PDF files using LlamaParse, handling both file paths and BytesIO objects
    
    Args:
        pdf_files (list): List of PDF file paths or BytesIO objects
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
    temp_files = []  # Keep track of temporary files to delete later

    try:
        for index, pdf_file in enumerate(pdf_files):
            try:
                # Check if pdf_file is a BytesIO object
                if hasattr(pdf_file, "read"):
                    # print(f"Processing BytesIO file {index + 1}/{len(pdf_files)}")
                    
                    # Create a temporary file
                    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    temp_path = temp.name
                    temp_files.append(temp_path)  # Add to list for cleanup
                    
                    # Reset pointer and write content to temp file
                    pdf_file.seek(0)
                    temp.write(pdf_file.read())
                    temp.close()
                    
                    # Parse the temporary file
                    document = parser.load_data(temp_path)
                    
                else:
                    # Handle as a regular file path
                    # print(f"Processing file {index + 1}/{len(pdf_files)}: {pdf_file}")
                    document = parser.load_data(pdf_file)
                
                documents.append(document)
                # print(f"Successfully processed file {index + 1}")
                
            except Exception as e:
                # print(f"Error processing file {index + 1}: {str(e)}")
                continue
                
        return documents
        
    finally:
        # Clean up temporary files
        for temp_path in temp_files:
            try:
                os.unlink(temp_path)
            except:
                pass