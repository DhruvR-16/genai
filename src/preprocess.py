import os
import sys
import json
import re
import pickle
from tqdm import tqdm
import spacy


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)


PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CLEAN_DOCS_PATH = os.path.join(PROCESSED_DIR, "clean_docs.pkl")
RAW_DOCS_PATH = os.path.join(PROCESSED_DIR, "raw_docs.pkl")


ARXIV_QUERIES = [
    "cat:cs.AI",              # Artificial Intelligence
    "cat:cs.CV",              # Computer Vision
    "cat:cs.LG",              # Machine Learning
    "cat:cs.CL",              # Computation and Language
    "cat:cs.CR",              # Cryptography and Security
    "cat:physics.gen-ph",     # General Physics
    "cat:quant-ph",           # Quantum Physics
    "cat:q-bio.BM",           # Biomolecules
    "cat:q-bio.NC",           # Neurons and Cognition
    "cat:econ.GN",            # General Economics
    "cat:q-fin.ST",           # Statistical Finance
    "cat:math.CO",            # Combinatorics
    "cat:math.PR",            # Probability
    "cat:stat.ML",            # Machine Learning (Stat)
    "cat:astro-ph.GA"         # Astrophysics of Galaxies
]
PAPERS_PER_QUERY = 70


ALLOWED_POS = ["NOUN", "PROPN", "ADJ"]
MIN_TOKEN_LENGTH = 3


nlp = spacy.load(
    "en_core_web_sm",
    disable=["parser", "ner"]
)

import arxiv

def load_arxiv_documents():
    documents = []
    seen_ids = set()
    
    for query in ARXIV_QUERIES:
        print(f"Fetching papers for query: {query}...")
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=PAPERS_PER_QUERY,
                sort_by=arxiv.SortCriterion.SubmittedDate
            )
            
            for result in client.results(search):
                if result.entry_id in seen_ids:
                    continue
                seen_ids.add(result.entry_id)

                title = result.title
                summary = result.summary
                text = f"{title}. {summary}"
                text = text.replace("\n", " ").strip()
                
                if text:
                    documents.append(text)
        except Exception as e:
            print(f"Error fetching for query {query}: {e}")
            
    return documents

def clean_noise(text):

    text = text.lower()

    # remove LaTeX commands and environments
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = re.sub(r"\\begin\{.*?\}.*?\\end\{.*?\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$]*\$", " ", text)

    # remove common LaTeX package names and artifacts
    latex_noise = [
        "usepackage", "documentclass", "amssymb", "amsmath", "amsfont",
        "wasysym", "amsbsy", "upgreek", "mathrsfs", "setlength",
        "minimal", "textwidth", "textheight", "oddsidemargin",
        "evensidemargin", "topmargin", "headheight", "headsep",
        "footskip", "columnsep", "pdfcreator", "pdftex",
        "linespread", "document", "begin", "end"
    ]
    for term in latex_noise:
        text = text.replace(term, " ")

    # remove citation numbers eg:=[12]
    text = re.sub(r"\[\d+\]", " ", text)

    # remove author citations eg:- (Smith et al., 2020)
    text = re.sub(r"\(.*?et al.,?\s*\d{4}\)", " ", text)

    # remove numbers
    text = re.sub(r"\d+", " ", text)

    # remove special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def preprocess_document(text):

    cleaned_text = clean_noise(text)

    doc = nlp(cleaned_text)

    tokens = []

    for token in doc:

        if token.is_stop:
            continue

        if token.is_punct:
            continue

        if token.pos_ not in ALLOWED_POS:
            continue

        lemma = token.lemma_.lower()

        if len(lemma) < MIN_TOKEN_LENGTH:
            continue

        tokens.append(lemma)

    return tokens

def preprocess_corpus(documents):

    processed_docs = []

    for doc in tqdm(documents, desc="Preprocessing Documents"):
        tokens = preprocess_document(doc)

        if len(tokens) > 0:
            processed_docs.append(tokens)

    return processed_docs

def save_processed_data(obj, output_path):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(obj, f)

    print(f"Saved: {output_path}")

def main():

    print("\nLoading ArXiv papers from multiple topics...")
    raw_documents = load_arxiv_documents()

    print(f"Loaded {len(raw_documents)} documents")

    save_processed_data(raw_documents, RAW_DOCS_PATH)

    print("\nStarting preprocessing...")
    clean_docs = preprocess_corpus(raw_documents)

    print(f"\nTotal cleaned documents: {len(clean_docs)}")

    avg_tokens = sum(len(doc) for doc in clean_docs) / len(clean_docs)
    print(f"Average tokens per document: {avg_tokens:.2f}")

    save_processed_data(clean_docs, CLEAN_DOCS_PATH)

    print("\nPreprocessing pipeline complete!")


if __name__ == "__main__":
    main()