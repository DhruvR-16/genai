import os
import sys
import pickle
import numpy as np


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)


PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

TFIDF_MATRIX_PATH = os.path.join(PROCESSED_DIR, "tfidf_matrix.pkl")
VECTORIZER_PATH = os.path.join(PROCESSED_DIR, "tfidf_vectorizer.pkl")
LDA_MODEL_PATH = os.path.join(PROCESSED_DIR, "lda_model.pkl")
DICTIONARY_PATH = os.path.join(PROCESSED_DIR, "dictionary.pkl")
KEYWORDS_PATH = os.path.join(PROCESSED_DIR, "keywords.pkl")

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_tfidf_keywords(tfidf_matrix, vectorizer, top_n=30):

    feature_names = vectorizer.get_feature_names_out()
    avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = avg_scores.argsort()[::-1][:top_n]

    keywords = [
        (feature_names[i], round(float(avg_scores[i]), 4))
        for i in top_indices
    ]

    return keywords

def get_topic_keywords(lda_model, num_words=10):

    topic_keywords = {}

    for topic_id in range(lda_model.num_topics):
        words = lda_model.show_topic(topic_id, topn=num_words)
        topic_keywords[topic_id] = [
            (word, round(float(weight), 4))
            for word, weight in words
        ]

    return topic_keywords

def main():

    print("\nLoading TF-IDF model...")
    tfidf_matrix = load_pickle(TFIDF_MATRIX_PATH)
    vectorizer = load_pickle(VECTORIZER_PATH)

    print("Loading LDA model...")
    lda_model = load_pickle(LDA_MODEL_PATH)

    print("Extracting global TF-IDF keywords...")
    global_keywords = get_tfidf_keywords(tfidf_matrix, vectorizer)

    print("Extracting per-topic keywords...")
    topic_keywords = get_topic_keywords(lda_model)

    keywords_data = {
        "global_tfidf": global_keywords,
        "topic_keywords": topic_keywords
    }

    os.makedirs(os.path.dirname(KEYWORDS_PATH), exist_ok=True)
    with open(KEYWORDS_PATH, "wb") as f:
        pickle.dump(keywords_data, f)

    print("\nGlobal Top 15 TF-IDF Keywords:")
    for word, score in global_keywords[:15]:
        print(f"  {word:30s} {score:.4f}")

    print("\nPer-Topic Keywords:")
    for topic_id, words in topic_keywords.items():
        top_words = ", ".join(w for w, _ in words[:5])
        print(f"  Topic {topic_id}: {top_words}")

    print("extraction complete")


if __name__ == "__main__":
    main()
