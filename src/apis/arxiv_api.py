import arxiv
import os
import json

RAW_DATA_PATH = "data/raw_data"

def fetch_papers(topic, max_results=10):
    client = arxiv.Client()
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH)

    papers = []
    for result in client.results(search):
        paper_data = {
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "summary": result.summary,
            "pdf_url": result.pdf_url
        }
        papers.append(paper_data)

        # Save each paper's content as a JSON file
        with open(os.path.join(RAW_DATA_PATH, f"{result.entry_id.split('/')[-1]}.json"), "w") as f:
            json.dump(paper_data, f, indent=4)

    print(f"Fetched {len(papers)} papers on '{topic}'")
    return papers
