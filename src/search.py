import os
import sys
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

TFIDF_MATRIX_PATH = os.path.join(PROJECT_ROOT, "data/processed/tfidf_matrix.pkl")
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "data/processed/tfidf_vectorizer.pkl")
DOCUMENTS_PATH = os.path.join(PROJECT_ROOT, "data/processed/raw_docs.pkl")

def load_objects():

    tfidf_matrix = pickle.load(
        open(TFIDF_MATRIX_PATH, "rb")
    )

    vectorizer = pickle.load(
        open(VECTORIZER_PATH, "rb")
    )

    documents = pickle.load(
        open(DOCUMENTS_PATH, "rb")
    )

    return tfidf_matrix, vectorizer, documents

def retrieve_documents(query, top_k=5):

    tfidf_matrix, vectorizer, documents = load_objects()

    # Convert query → TF-IDF vector
    query_vector = vectorizer.transform([query])

    # Compute similarity
    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    top_indices = np.argsort(similarities)[::-1][:top_k]

    retrieved_docs = [
        documents[i]
        for i in top_indices
    ]

    scores = [
        similarities[i]
        for i in top_indices
    ]

    return retrieved_docs, scores

if __name__ == "__main__":

    query = "machine learning healthcare"

    docs, scores = retrieve_documents(query)

    for i, score in enumerate(scores):
        print(f"\nDocument {i+1} | Score: {score:.3f}")
        print(docs[i][:400])