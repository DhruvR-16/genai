import os
import sys
import pickle
from collections import Counter


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)


PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

RAW_DOCS_PATH = os.path.join(PROCESSED_DIR, "raw_docs.pkl")
CLEAN_DOCS_PATH = os.path.join(PROCESSED_DIR, "clean_docs.pkl")
EDA_STATS_PATH = os.path.join(PROCESSED_DIR, "eda_stats.pkl")

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def compute_stats(raw_docs, clean_docs):

    raw_lengths = [len(doc.split()) for doc in raw_docs]
    avg_raw_length = sum(raw_lengths) / len(raw_lengths)

    clean_lengths = [len(doc) for doc in clean_docs]
    avg_clean_length = sum(clean_lengths) / len(clean_lengths)

    # vocabulary
    all_tokens = [token for doc in clean_docs for token in doc]
    vocab = set(all_tokens)
    total_tokens = len(all_tokens)

    # token frequency distribution
    token_freq = Counter(all_tokens)
    top_tokens = token_freq.most_common(50)

    stats = {
        "num_documents": len(raw_docs),
        "avg_raw_word_count": round(avg_raw_length, 1),
        "avg_clean_token_count": round(avg_clean_length, 1),
        "vocabulary_size": len(vocab),
        "total_tokens": total_tokens,
        "top_50_tokens": top_tokens,
        "raw_doc_lengths": raw_lengths,
        "clean_doc_lengths": clean_lengths
    }

    return stats

def main():

    print("\nLoading data")
    raw_docs = load_pickle(RAW_DOCS_PATH)
    clean_docs = load_pickle(CLEAN_DOCS_PATH)

    print("Computing EDA statistics...")
    stats = compute_stats(raw_docs, clean_docs)

    os.makedirs(os.path.dirname(EDA_STATS_PATH), exist_ok=True)
    with open(EDA_STATS_PATH, "wb") as f:
        pickle.dump(stats, f)

    print(f"\nCorpus Statistics:")
    print(f"  Documents:          {stats['num_documents']}")
    print(f"  Avg raw words:      {stats['avg_raw_word_count']}")
    print(f"  Avg clean tokens:   {stats['avg_clean_token_count']}")
    print(f"  Vocabulary size:    {stats['vocabulary_size']}")
    print(f"  Total tokens:       {stats['total_tokens']}")


if __name__ == "__main__":
    main()
