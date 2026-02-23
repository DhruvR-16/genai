import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from src.search import retrieve_documents


from src.textrank_summary import textrank_summary

def summarize_query(query, top_k_docs=5):

    documents, scores = retrieve_documents(
        query,
        top_k=top_k_docs
    )

    combined_text = " ".join(documents)

    summary = textrank_summary(combined_text, top_n=5)

    return summary, scores

if __name__ == "__main__":

    query = "machine learning healthcare"

    summary, scores = summarize_query(query)

    print("\nResearch Summary:\n")

    for s in summary:
        print("-", s)