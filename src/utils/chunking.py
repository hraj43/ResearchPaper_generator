def chunk_text(text, chunk_size=512, overlap=50):
    """
    Splits text into overlapping chunks of specified size.

    Args:
        text (str): The text to be chunked.
        chunk_size (int): Maximum size of each chunk.
        overlap (int): Number of overlapping words between chunks.

    Returns:
        list: List of text chunks.
    """
    words = text.split()  # Split text into words
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])  # Form chunk
        chunks.append(chunk)

    return chunks
