from apis.arxiv_api import fetch_papers  # Step 1: Data Extraction
from utils.data_cleaner import clean_text
from utils.chunking import chunk_text

def main():
    query = "Artificial Intelligence"
    papers = fetch_papers(query)

    # Extract content from fetched papers
    all_text = " ".join(paper['summary'] for paper in papers)

    # Step 2: Data Cleaning
    cleaned_text = clean_text(all_text)
    print(f"Cleaned Text:\n{cleaned_text[:500]}...\n")

    # Step 3: Chunking
    chunks = chunk_text(cleaned_text)
    for idx, chunk in enumerate(chunks[:5]):  # Displaying first 5 chunks
        print(f"Chunk {idx + 1}:\n{chunk}\n"," ")

if __name__ == "__main__":
    main()
