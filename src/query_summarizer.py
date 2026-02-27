import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from src.search import retrieve_documents


from src.textrank_summary import textrank_summary

def summarize_query(query, top_k_docs=5, sentences_per_doc=2):
    
    documents, scores = retrieve_documents(
        query,
        top_k=top_k_docs
    )
    
    summary_items = []
    
    for i, doc in enumerate(documents):
        # Extract sentences specifically from this paper
        doc_sentences = textrank_summary(doc, top_n=sentences_per_doc)
        
        for sent in doc_sentences:
            summary_items.append({
                "rank": i + 1,
                "score": scores[i],
                "sentence": sent
            })
            
    return summary_items, scores

if __name__ == "__main__":
    
    query = "machine learning healthcare"
    
    summary_items, scores = summarize_query(query)
    
    print("\nResearch Highlights (Extractive):\n")
    
    for item in summary_items:
        print(f"[Paper Rank {item['rank']}] - {item['sentence']}")