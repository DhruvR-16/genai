import os
import sys
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)


PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

INPUT_PATH = os.path.join(PROCESSED_DIR, "clean_docs.pkl")
TFIDF_MATRIX_PATH = os.path.join(PROCESSED_DIR, "tfidf_matrix.pkl")
VECTORIZER_PATH = os.path.join(PROCESSED_DIR, "tfidf_vectorizer.pkl")
TOP_KEYWORDS_PATH = os.path.join(PROCESSED_DIR, "top_keywords.pkl")

def load_clean_corpus(path):

    with open(path, "rb") as f:
        clean_docs = pickle.load(f)

    return clean_docs

def join_tokens(clean_docs):

    documents = [" ".join(doc) for doc in clean_docs]

    return documents

def build_tfidf(documents):

    vectorizer = TfidfVectorizer(
        max_df=0.85,
        min_df=2,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    return tfidf_matrix, vectorizer

def extract_top_keywords(tfidf_matrix, vectorizer, top_n=30):

    feature_names = vectorizer.get_feature_names_out()

    # Average TF-IDF score across all documents
    avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

    top_indices = avg_scores.argsort()[::-1][:top_n]

    top_keywords = [
        (feature_names[i], avg_scores[i])
        for i in top_indices
    ]

    return top_keywords

def save_object(obj, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)

def main():

    print("\nLoading cleaned corpus...")
    clean_docs = load_clean_corpus(INPUT_PATH)

    print("Preparing documents...")
    documents = join_tokens(clean_docs)

    print("Building TF-IDF representation...")
    tfidf_matrix, vectorizer = build_tfidf(documents)

    print("Extracting top keywords...")
    top_keywords = extract_top_keywords(tfidf_matrix, vectorizer)

    print("Saving TF-IDF outputs...")
    save_object(tfidf_matrix, TFIDF_MATRIX_PATH)
    save_object(vectorizer, VECTORIZER_PATH)
    save_object(top_keywords, TOP_KEYWORDS_PATH)

    print("\nTF-IDF completed successfully")
    print("Matrix shape:", tfidf_matrix.shape)

    print("\nTop 15 Keywords:")
    for word, score in top_keywords[:15]:
        print(f"  {word:30s} {score:.4f}")


if __name__ == "__main__":
    main()