import os
import sys
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from src.search import retrieve_documents
from src.query_summarizer import summarize_query

# =====================================================
# HELPER: Load pickle
# =====================================================

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

def load_pkl(filename):
    path = os.path.join(PROCESSED_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)

st.set_page_config(
    page_title="Research Topic Analyzer",
    page_icon="📚",
    layout="wide"
)


st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 0.5rem 0 0.2rem 0;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-card p {
        margin: 0.3rem 0 0 0;
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .topic-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
    }
    .keyword-badge {
        display: inline-block;
        background: #e8eaf6;
        color: #3f51b5;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .result-card {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .score-badge {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATA (cached)
# =====================================================

@st.cache_resource
def load_all_data():
    data = {}
    data["eda_stats"] = load_pkl("eda_stats.pkl")
    data["clean_docs"] = load_pkl("clean_docs.pkl")
    data["lda_model"] = load_pkl("lda_model.pkl")
    data["coherence"] = load_pkl("coherence_scores.pkl")
    data["keywords"] = load_pkl("keywords.pkl")
    data["summaries"] = load_pkl("summaries.pkl")
    return data

data = load_all_data()

st.markdown('<div class="main-header">📚 Intelligent Research Topic Analysis System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">NLP-powered analysis of ArXiv research papers — Preprocessing, Topic Modeling, Keyword Extraction & Summarization</div>', unsafe_allow_html=True)


tab4, tab1, tab2, tab3 = st.tabs([
    "🔍 Search & Summary",
    "📊 EDA & Preprocessing",
    "🧠 Topic Modeling",
    "🔑 Keyword Extraction"
])

with tab1:
    st.header("Exploratory Data Analysis")

    stats = data["eda_stats"]

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{stats['num_documents']}</h3>
            <p>Research Papers</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3>{stats['avg_raw_word_count']}</h3>
            <p>Avg Words / Doc</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3>{stats['vocabulary_size']:,}</h3>
            <p>Vocabulary Size</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <h3>{stats['total_tokens']:,}</h3>
            <p>Total Tokens</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📈 Top 30 Token Frequencies")
        top_tokens = stats["top_50_tokens"][:30]
        words = [t[0] for t in top_tokens]
        counts = [t[1] for t in top_tokens]

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(words)))
        ax.barh(words[::-1], counts[::-1], color=colors)
        ax.set_xlabel("Frequency", fontsize=12)
        ax.set_title("Most Frequent Tokens in Corpus", fontsize=14, fontweight="bold")
        ax.tick_params(axis='y', labelsize=10)
        plt.tight_layout()
        st.pyplot(fig)

    with col_right:
        st.subheader("☁️ Word Cloud")
        all_tokens = [token for doc in data["clean_docs"] for token in doc]
        token_freq = Counter(all_tokens)

        wc = WordCloud(
            width=800,
            height=600,
            background_color="white",
            colormap="viridis",
            max_words=100,
            contour_width=2,
            contour_color="#667eea"
        ).generate_from_frequencies(token_freq)

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        ax2.imshow(wc, interpolation="bilinear")
        ax2.axis("off")
        plt.tight_layout()
        st.pyplot(fig2)

    # Document length distribution
    st.subheader("📏 Document Length Distribution")
    col_a, col_b = st.columns(2)

    with col_a:
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.hist(stats["raw_doc_lengths"], bins=20, color="#667eea", edgecolor="white", alpha=0.85)
        ax3.set_xlabel("Word Count")
        ax3.set_ylabel("Number of Documents")
        ax3.set_title("Raw Document Lengths", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig3)

    with col_b:
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        ax4.hist(stats["clean_doc_lengths"], bins=20, color="#f5576c", edgecolor="white", alpha=0.85)
        ax4.set_xlabel("Token Count")
        ax4.set_ylabel("Number of Documents")
        ax4.set_title("Cleaned Document Lengths (After Preprocessing)", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig4)

with tab2:
    st.header("LDA Topic Modeling Results")

    lda_model = data["lda_model"]
    coherence_data = data["coherence"]

    # Coherence info
    best_k = coherence_data["best_k"]
    scores = coherence_data["scores"]
    best_score = scores[best_k]["coherence"]
    best_perp = scores[best_k]["perplexity"]

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);">
            <h3>{best_k} Topics</h3>
            <p>Optimal Number</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);">
            <h3>{best_score:.4f}</h3>
            <p>Coherence (c_v)</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); margin-top: 1rem;">
            <h3>{best_perp:.2f}</h3>
            <p>Log Perplexity</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

    with col3:
        st.info("💡 **Why no R² or Recall?**\nR² and Recall require labeled data (Supervised Learning). Since we are dynamically discovering hidden topics in raw research papers (Unsupervised Learning), we evaluate model quality using **Coherence** (how semantically similar the top words are) and **Perplexity** (how well the model predicts new data).")

    with col3:
        st.subheader("📉 Coherence Optimization")
        ks = sorted(scores.keys())
        vals = [scores[k]["coherence"] for k in ks]

        fig5, ax5 = plt.subplots(figsize=(6, 3))
        ax5.plot(ks, vals, "o-", color="#667eea", linewidth=2, markersize=8)
        ax5.axvline(x=best_k, color="#f5576c", linestyle="--", alpha=0.7, label=f"Best k={best_k}")
        ax5.set_xlabel("Number of Topics", fontsize=10)
        ax5.set_ylabel("Coherence Score", fontsize=10)
        ax5.legend()
        ax5.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig5)

    st.markdown("---")

    # Display topics
    st.subheader("📋 Discovered Topics")

    topic_keywords = data["keywords"]["topic_keywords"]

    for topic_id in range(lda_model.num_topics):
        words = topic_keywords.get(topic_id, [])
        keyword_badges = " ".join(
            f'<span class="keyword-badge">{w} ({s:.3f})</span>'
            for w, s in words[:8]
        )
        st.markdown(f"""
        <div class="topic-card">
            <strong>Topic {topic_id + 1}</strong><br>
            {keyword_badges}
        </div>
        """, unsafe_allow_html=True)

    # Per-topic keyword charts
    st.subheader("📊 Topic Keyword Weights")

    num_topics = lda_model.num_topics
    cols_per_row = min(3, num_topics)
    rows_needed = (num_topics + cols_per_row - 1) // cols_per_row

    for row in range(rows_needed):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            topic_id = row * cols_per_row + col_idx
            if topic_id >= num_topics:
                break

            with cols[col_idx]:
                words = topic_keywords.get(topic_id, [])
                w_names = [w for w, _ in words[:8]]
                w_scores = [s for _, s in words[:8]]

                fig_t, ax_t = plt.subplots(figsize=(5, 3))
                ax_t.barh(w_names[::-1], w_scores[::-1],
                         color=plt.cm.Set2(np.linspace(0, 1, len(w_names))))
                ax_t.set_title(f"Topic {topic_id + 1}", fontweight="bold", fontsize=11)
                ax_t.tick_params(axis='y', labelsize=9)
                plt.tight_layout()
                st.pyplot(fig_t)

with tab3:
    st.header("Keyword Extraction Results")

    keywords_data = data["keywords"]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔤 Top Global TF-IDF Keywords")

        global_kw = keywords_data["global_tfidf"]

        # Table
        st.markdown("| Rank | Keyword | TF-IDF Score |")
        st.markdown("|------|---------|-------------|")
        for i, (word, score) in enumerate(global_kw[:20]):
            st.markdown(f"| {i+1} | **{word}** | {score:.4f} |")

    with col2:
        st.subheader("☁️ TF-IDF Keyword Cloud")

        kw_freq = {w: s for w, s in global_kw}
        wc2 = WordCloud(
            width=700,
            height=500,
            background_color="white",
            colormap="plasma",
            max_words=50
        ).generate_from_frequencies(kw_freq)

        fig6, ax6 = plt.subplots(figsize=(8, 6))
        ax6.imshow(wc2, interpolation="bilinear")
        ax6.axis("off")
        plt.tight_layout()
        st.pyplot(fig6)

    st.markdown("---")

    st.subheader("🏷️ Keywords by Topic")

    topic_kw = keywords_data["topic_keywords"]
    for topic_id, words in topic_kw.items():
        badges = " ".join(
            f'<span class="keyword-badge">{w}</span>' for w, _ in words[:10]
        )
        st.markdown(f"""
        <div class="topic-card">
            <strong>Topic {topic_id + 1}</strong><br>
            {badges}
        </div>
        """, unsafe_allow_html=True)

with tab4:
    st.header("Document Search & Summarization")

    query = st.text_input(
        "🔎 Enter a research query",
        placeholder="e.g. virus treatment, machine learning healthcare, vaccine development"
    )

    if st.button("🚀 Analyze", type="primary"):

        if not query.strip():
            st.warning("Please enter a research query.")
            st.stop()

        with st.spinner("Retrieving relevant documents..."):
            docs, scores = retrieve_documents(query, top_k=5)

        st.subheader("📄 Retrieved Papers")

        for i, (doc, score) in enumerate(zip(docs, scores)):
            preview = doc[:300] + "..." if len(doc) > 300 else doc
            st.markdown(f"""
            <div class="result-card">
                <strong>Paper {i+1}</strong>
                <span class="score-badge">Score: {score:.4f}</span>
                <br><br>
                <span style="color:#555; font-size:0.9rem;">{preview}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        with st.spinner("Generating extractive summary..."):
            summary, _ = summarize_query(query, top_k_docs=5)

        st.subheader("📝 Extractive Research Summary")

        if summary:
            for i, sentence in enumerate(summary):
                st.markdown(f"**{i+1}.** {sentence}")
        else:
            st.info("No summary could be generated for this query.")

with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This system performs **traditional NLP analysis**
    on ArXiv research papers.

    **Pipeline:**
    1. Text Preprocessing (spaCy)
    2. TF-IDF Feature Extraction
    3. LDA Topic Modeling
    4. TextRank Summarization
    5. Cosine Similarity Retrieval

    **Dataset:** ArXiv (100 papers from diverse topics)
    """)

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    - Python, spaCy, NLTK
    - scikit-learn (TF-IDF)
    - Gensim (LDA)
    - NetworkX (TextRank)
    - Streamlit (UI)
    """)

    st.markdown("---")
    stats = data["eda_stats"]
    st.markdown(f"**📊 Corpus:** {stats['num_documents']} papers")
    st.markdown(f"**📖 Vocab:** {stats['vocabulary_size']:,} terms")