import arxiv
import argparse
import json
from rake_nltk import Rake
import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download('punkt_tab')

def refine_query(query):
    rake = Rake()
    rake.extract_keywords_from_text(query)
    keywords = rake.get_ranked_phrases()
    return " ".join(keywords)

def scrape_papers(query, numresults) -> list:
    refined_query = refine_query(query)
    results = []

    search = arxiv.Search(
        query=refined_query,
        max_results=numresults,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = list(search.results())

    for i, p in enumerate(papers):
        paper_doc = {
            "url": p.pdf_url,
            "title": p.title, 
            "abstract": p.summary
        }
        results.append(paper_doc)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="the query to search for", type=str)
    parser.add_argument(
        "--numresults", help="the number of results to return", type=int
    )
    args = parser.parse_args()

    results = scrape_papers(**args)
    for i, r in enumerate(results):
        with open(f"src/data/data_{i}.json", "w") as f:
            json.dump(r, f)
