import os
import sys
import pickle
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)


PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

INPUT_PATH = os.path.join(PROCESSED_DIR, "clean_docs.pkl")
DICTIONARY_PATH = os.path.join(PROCESSED_DIR, "dictionary.pkl")
CORPUS_PATH = os.path.join(PROCESSED_DIR, "corpus.pkl")
LDA_MODEL_PATH = os.path.join(PROCESSED_DIR, "lda_model.pkl")
COHERENCE_PATH = os.path.join(PROCESSED_DIR, "coherence_scores.pkl")

def load_clean_docs(path):

    with open(path, "rb") as f:
        docs = pickle.load(f)

    return docs


def build_corpus(clean_docs):

    dictionary = corpora.Dictionary(clean_docs)

    dictionary.filter_extremes(
        no_below=2,
        no_above=0.85
    )

    corpus = [
        dictionary.doc2bow(doc)
        for doc in clean_docs
    ]

    return dictionary, corpus

def train_lda(corpus, dictionary, num_topics):

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=15,
        random_state=42
    )

    return lda_model

def compute_coherence(model, texts, dictionary):

    coherence_model = CoherenceModel(
        model=model,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v"
    )

    return coherence_model.get_coherence()

def optimize_topics(corpus, dictionary, texts):

    best_score = 0
    best_model = None
    best_k = 2
    coherence_scores = {}

    for k in range(2, 11):

        print(f"\nTraining LDA with {k} topics")

        model = train_lda(corpus, dictionary, k)
        score = compute_coherence(model, texts, dictionary)
        perplexity = model.log_perplexity(corpus)

        coherence_scores[k] = {
            "coherence": round(score, 4),
            "perplexity": round(perplexity, 4)
        }
        print(f"Coherence: {score:.4f} | Perplexity: {perplexity:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_k = k

    print(f"\nBest: {best_k} topics (coherence={best_score:.4f})")

    return best_model, coherence_scores, best_k

def save_object(obj, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)

def main():

    print("\nLoading cleaned documents...")
    clean_docs = load_clean_docs(INPUT_PATH)

    print("Building corpus...")
    dictionary, corpus = build_corpus(clean_docs)

    print("Optimizing topic count...")
    lda_model, coherence_scores, best_k = optimize_topics(
        corpus,
        dictionary,
        clean_docs
    )

    print("\nSaving outputs...")
    save_object(dictionary, DICTIONARY_PATH)
    save_object(corpus, CORPUS_PATH)
    save_object(lda_model, LDA_MODEL_PATH)
    save_object({
        "scores": coherence_scores,
        "best_k": best_k
    }, COHERENCE_PATH)

    print("\nTopic Modeling Complete")

    print("\nTop Topics:")
    for topic in lda_model.print_topics():
        print(topic)


if __name__ == "__main__":
    main()