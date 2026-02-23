import os
import sys
import pickle
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)


PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RAW_DOCS_PATH = os.path.join(PROCESSED_DIR, "raw_docs.pkl")
SUMMARY_OUTPUT = os.path.join(PROCESSED_DIR, "summaries.pkl")

nlp = spacy.load("en_core_web_sm")

def load_raw_documents():

    with open(RAW_DOCS_PATH, "rb") as f:
        documents = pickle.load(f)

    return documents

def split_sentences(text):

    doc = nlp(text)

    sentences = [
        sent.text.strip()
        for sent in doc.sents
        if len(sent.text.split()) > 8
    ]

    return sentences

def build_similarity(sentences):

    vectorizer = TfidfVectorizer()

    tfidf = vectorizer.fit_transform(sentences)

    similarity_matrix = (
        tfidf * tfidf.T
    ).toarray()

    return similarity_matrix

def textrank_summary(text, top_n=5):

    sentences = split_sentences(text)

    if len(sentences) < top_n:
        return sentences

    sim_matrix = build_similarity(sentences)

    graph = nx.from_numpy_array(sim_matrix)

    scores = nx.pagerank(graph)

    ranked_sentences = sorted(
        ((scores[i], s)
         for i, s in enumerate(sentences)),
        reverse=True
    )

    summary = [
        sent for _, sent
        in ranked_sentences[:top_n]
    ]

    return summary

def summarize_corpus(documents):

    summaries = []

    for i, doc in enumerate(documents):
        print(f"  Summarizing document {i+1}/{len(documents)}...")
        summary = textrank_summary(doc)
        summaries.append(summary)

    return summaries

def save_summaries(data, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(data, f)

def main():

    print("Loading raw documents...")
    docs = load_raw_documents()
    print(f"Loaded {len(docs)} documents")

    print("Generating summaries...")
    summaries = summarize_corpus(docs)

    save_summaries(summaries, SUMMARY_OUTPUT)

    print("Summaries saved successfully!")


if __name__ == "__main__":
    main()