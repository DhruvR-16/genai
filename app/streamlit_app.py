import os
import sys
import subprocess
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

# ── Patch numpy BitGenerator unpickling for pickles saved with older numpy ──
try:
    import numpy.random._pickle as _np_rng_pickle
    from numpy.random import MT19937, PCG64, SFC64, Philox
    _CLASS_TO_NAME = {
        MT19937: "MT19937",
        PCG64:   "PCG64",
        SFC64:   "SFC64",
        Philox:  "Philox",
    }
    _original_ctor = _np_rng_pickle.__bit_generator_ctor

    def _patched_ctor(bit_generator_name):
        # Old numpy pickled the class object; new numpy expects a string name
        if not isinstance(bit_generator_name, str):
            bit_generator_name = _CLASS_TO_NAME.get(
                bit_generator_name,
                getattr(bit_generator_name, "__name__", str(bit_generator_name))
            )
        return _original_ctor(bit_generator_name)

    _np_rng_pickle.__bit_generator_ctor = _patched_ctor
except Exception:
    pass  # Non-critical; worst case the user needs to re-run the pipeline

# ── Ensure spaCy model is available before any src import tries to load it ──
def _ensure_spacy_model(model: str = "en_core_web_sm"):
    try:
        import spacy
        spacy.load(model)
    except OSError:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model],
            check=True,
            capture_output=True,
        )

_ensure_spacy_model()

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

# =====================================================
# GLOBAL STYLES
# =====================================================

st.markdown("""
<style>
    /* ---- Global typography ---- */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: var(--text-color);
        text-align: center;
        padding: 0.6rem 0 0.2rem 0;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: var(--text-color);
        opacity: 0.8;
        text-align: center;
        margin-bottom: 1.8rem;
    }

    /* ---- Metric cards ---- */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 14px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    .metric-card h3 { margin: 0; font-size: 2rem; font-weight: 700; }
    .metric-card p  { margin: 0.3rem 0 0 0; font-size: 0.85rem; opacity: 0.9; }

    /* ---- Topic cards ---- */
    .topic-card {
        background: var(--secondary-background-color);
        border-left: 5px solid #667eea;
        padding: 1rem 1.2rem;
        border-radius: 0 10px 10px 0;
        margin-bottom: 0.9rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* ---- Keyword badge ---- */
    .keyword-badge {
        display: inline-block;
        background: var(--primary-color);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.82rem;
        font-weight: 500;
    }

    /* ---- Result cards with rank color coding ---- */
    .result-card {
        background: var(--secondary-background-color);
        border: 1px solid var(--secondary-background-color);
        border-radius: 12px;
        padding: 1rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        transition: box-shadow 0.2s;
    }
    .result-card:hover { box-shadow: 0 4px 18px rgba(0,0,0,0.12); }

    .rank-1 { border-left: 5px solid #FFD700; }
    .rank-2 { border-left: 5px solid #C0C0C0; }
    .rank-3 { border-left: 5px solid #CD7F32; }
    .rank-other { border-left: 5px solid #667eea; }

    /* ---- Score badges ---- */
    .score-badge {
        background: rgba(46, 125, 50, 0.2);
        color: #2e7d32;
        padding: 0.2rem 0.7rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .tfidf-badge {
        background: #e3f2fd;
        color: #1565c0;
        padding: 0.2rem 0.7rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .bm25-badge {
        background: #fce4ec;
        color: #b71c1c;
        padding: 0.2rem 0.7rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }

    /* ---- Summary sentences ---- */
    .summary-item {
        background: var(--secondary-background-color);
        border-left: 4px solid #66bb6a;
        padding: 0.7rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.6rem;
        font-size: 0.93rem;
        line-height: 1.6;
        color: var(--text-color);
    }
    .summary-num {
        font-weight: 700;
        color: #2e7d32;
        margin-right: 0.4rem;
    }

    /* ---- Timing pill ---- */
    .timing-pill {
        display: inline-block;
        background: #ede7f6;
        color: #4527a0;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }

    /* ---- Keyword rank row ---- */
    .kw-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.35rem;
        font-size: 0.88rem;
    }
    .kw-rank {
        min-width: 2rem;
        font-weight: 700;
        color: var(--text-color);
        opacity: 0.8;
    }
    .kw-name { flex: 1; font-weight: 500; }
    .kw-score { min-width: 4rem; text-align: right; color: var(--text-color); opacity: 0.6; font-size: 0.8rem; }
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

# =====================================================
# HEADER
# =====================================================

st.markdown('<div class="main-header">📚 Intelligent Research Topic Analysis System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">NLP-powered analysis of ArXiv research papers — '
    'TF-IDF Retrieval · TextRank Summarization · LDA Topic Modeling · Keyword Extraction</div>',
    unsafe_allow_html=True
)

# =====================================================
# TABS
# =====================================================

tab4, tab1, tab2, tab3 = st.tabs([
    "🔍 Search & Summary",
    "📊 EDA & Preprocessing",
    "🧠 Topic Modeling",
    "🔑 Keyword Extraction"
])

# ───────────────────────────────────────────────────────────────────────────
# TAB 1 — EDA
# ───────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("Exploratory Data Analysis")

    stats = data["eda_stats"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{stats['num_documents']:,}</h3>
            <p>Research Papers</p>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3>{stats['avg_raw_word_count']}</h3>
            <p>Avg Words / Doc</p>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3>{stats['vocabulary_size']:,}</h3>
            <p>Vocabulary Size</p>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <h3>{stats['total_tokens']:,}</h3>
            <p>Total Tokens</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📈 Top 30 Token Frequencies")
        top_tokens = stats["top_50_tokens"][:30]
        words  = [t[0] for t in top_tokens]
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
        all_tokens  = [token for doc in data["clean_docs"] for token in doc]
        token_freq  = Counter(all_tokens)

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

    st.subheader("📏 Document Length Distribution")
    col_a, col_b = st.columns(2)

    with col_a:
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.hist(stats["raw_doc_lengths"], bins=25, color="#667eea", edgecolor="white", alpha=0.85)
        ax3.set_xlabel("Word Count")
        ax3.set_ylabel("Number of Documents")
        ax3.set_title("Raw Document Lengths", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig3)

    with col_b:
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        ax4.hist(stats["clean_doc_lengths"], bins=25, color="#f5576c", edgecolor="white", alpha=0.85)
        ax4.set_xlabel("Token Count")
        ax4.set_ylabel("Number of Documents")
        ax4.set_title("Cleaned Document Lengths (After Preprocessing)", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig4)

# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — TOPIC MODELING
# ───────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("LDA Topic Modeling Results")

    lda_model      = data["lda_model"]
    coherence_data = data["coherence"]

    best_k     = coherence_data["best_k"]
    scores     = coherence_data["scores"]
    best_score = scores[best_k]["coherence"]
    best_perp  = scores[best_k]["perplexity"]

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg,#a18cd1,#fbc2eb);">
            <h3>{best_k}</h3><p>Optimal Topics</p>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg,#84fab0,#8fd3f4);">
            <h3>{best_score:.4f}</h3><p>Coherence (c_v)</p>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg,#ff9a9e,#fecfef);">
            <h3>{best_perp:.2f}</h3><p>Log Perplexity</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.info(
        "💡 **Evaluation metrics for unsupervised topic models** — "
        "**Coherence (c_v)** measures how semantically similar the top words in each topic are "
        "(higher = more coherent). **Perplexity** measures how well the model predicts held-out "
        "documents (lower absolute value = better fit). R² and Recall require labelled data and "
        "are not applicable here."
    )

    st.markdown("---")

    # Coherence curve
    st.subheader("📉 Coherence Optimisation Curve")
    ks   = sorted(scores.keys())
    vals = [scores[k]["coherence"] for k in ks]

    fig5, ax5 = plt.subplots(figsize=(9, 3))
    ax5.plot(ks, vals, "o-", color="#667eea", linewidth=2.5, markersize=9)
    ax5.fill_between(ks, vals, alpha=0.12, color="#667eea")
    ax5.axvline(x=best_k, color="#f5576c", linestyle="--", alpha=0.8,
                label=f"Best k = {best_k}  (coherence {best_score:.4f})")
    ax5.set_xlabel("Number of Topics", fontsize=11)
    ax5.set_ylabel("Coherence Score", fontsize=11)
    ax5.set_title("LDA Coherence vs. Number of Topics", fontweight="bold")
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig5)

    st.markdown("---")

    # Topic cards with rank
    st.subheader("📋 Discovered Topics — Ranked by Topic ID")
    topic_keywords = data["keywords"]["topic_keywords"]

    for topic_id in range(lda_model.num_topics):
        words   = topic_keywords.get(topic_id, [])
        badges  = " ".join(
            f'<span class="keyword-badge">#{j+1} {w} <em>({s:.3f})</em></span>'
            for j, (w, s) in enumerate(words[:8])
        )
        st.markdown(f"""
        <div class="topic-card">
            <strong>🗂 Topic {topic_id + 1}</strong>
            &nbsp;&nbsp;<small style="opacity:0.7;">Top 8 keywords by LDA weight</small><br><br>
            {badges}
        </div>""", unsafe_allow_html=True)

    # Per-topic keyword charts
    st.subheader("📊 Topic Keyword Weights")
    num_topics    = lda_model.num_topics
    cols_per_row  = min(3, num_topics)
    rows_needed   = (num_topics + cols_per_row - 1) // cols_per_row

    for row in range(rows_needed):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            topic_id = row * cols_per_row + col_idx
            if topic_id >= num_topics:
                break
            with cols[col_idx]:
                words    = topic_keywords.get(topic_id, [])
                w_names  = [f"#{j+1} {w}" for j, (w, _) in enumerate(words[:8])]
                w_scores = [s for _, s in words[:8]]

                fig_t, ax_t = plt.subplots(figsize=(5, 3.2))
                bars = ax_t.barh(w_names[::-1], w_scores[::-1],
                                 color=plt.cm.Set2(np.linspace(0, 1, len(w_names))))
                ax_t.set_title(f"Topic {topic_id + 1}", fontweight="bold", fontsize=11)
                ax_t.tick_params(axis='y', labelsize=8)
                ax_t.set_xlabel("LDA Weight", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig_t)

# ───────────────────────────────────────────────────────────────────────────
# TAB 3 — KEYWORD EXTRACTION
# ───────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("Keyword Extraction Results")

    keywords_data = data["keywords"]
    global_kw     = keywords_data["global_tfidf"]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔤 Top 20 Global TF-IDF Keywords")
        max_score = global_kw[0][1] if global_kw else 1

        for i, (word, score) in enumerate(global_kw[:20]):
            medal = {0: "🥇", 1: "🥈", 2: "🥉"}.get(i, f"#{i+1}")
            norm  = score / max_score
            st.markdown(
                f"**{medal} {word}** — `{score:.4f}`",
                unsafe_allow_html=False
            )
            st.progress(float(norm))

    with col2:
        st.subheader("☁️ TF-IDF Keyword Cloud")
        kw_freq = {w: s for w, s in global_kw}
        wc2 = WordCloud(
            width=700, height=500,
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
    st.subheader("🏷️ Keywords by Topic (Top 10 each)")

    topic_kw = keywords_data["topic_keywords"]
    for topic_id, words in topic_kw.items():
        top_words  = words[:10]
        max_w_score = top_words[0][1] if top_words else 1

        with st.expander(f"🗂 Topic {topic_id + 1} — {', '.join(w for w, _ in top_words[:4])} …"):
            for j, (w, s) in enumerate(top_words):
                medal = {0: "🥇", 1: "🥈", 2: "🥉"}.get(j, f"#{j+1}")
                st.markdown(f"**{medal} {w}** — `{s:.4f}`")
                st.progress(float(s / max_w_score))

# ───────────────────────────────────────────────────────────────────────────
# TAB 4 — SEARCH & SUMMARY
# ───────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("Document Search & Summarization")

    with st.sidebar:
        st.markdown("### ⚙️ Search Settings")
        top_k_slider = st.slider("Number of results", min_value=3, max_value=10, value=5, step=1)
        summary_n = st.slider("Extracted sentences per paper", min_value=1, max_value=4, value=2, step=1)

    query = st.text_input(
        "🔎 Enter a research query",
        placeholder="e.g. deep learning drug discovery, quantum computing optimization, climate neural networks"
    )

    col_btn, col_tip = st.columns([1, 4])
    with col_btn:
        search_clicked = st.button("🚀 Analyze", type="primary")
    with col_tip:
        st.caption("Hint: use domain-specific terms for better results, e.g. 'transformer attention mechanism' or 'CRISPR gene editing'")

    if search_clicked:

        if not query.strip():
            st.warning("Please enter a research query.")
            st.stop()

        # ── Retrieval ──
        with st.spinner("Running TF-IDF retrieval…"):
            docs, scores = retrieve_documents(query, top_k=top_k_slider)

        st.success(f"✅ Retrieved {len(docs)} papers using TF-IDF")

        # Score distribution bar chart
        st.subheader("📊 Retrieval Score Overview")
        fig_sc, ax_sc = plt.subplots(figsize=(8, 2.5))
        colors_sc = ["#FFD700" if i == 0 else "#C0C0C0" if i == 1 else "#CD7F32" if i == 2 else "#667eea"
                     for i in range(len(scores))]
        bars_sc = ax_sc.barh(
            [f"#{i+1}" for i in range(len(docs))],
            scores,
            color=colors_sc,
            edgecolor="white"
        )
        ax_sc.set_xlabel("TF-IDF Score", fontsize=10)
        ax_sc.set_title("Retrieved Paper Rankings", fontweight="bold")
        ax_sc.set_xlim(0, 1)
        for bar, s in zip(bars_sc, scores):
            ax_sc.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                       f"{s:.3f}", va="center", fontsize=9, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_sc)

        st.markdown("---")
        st.subheader("📄 Retrieved Papers")

        medals = {0: "🥇", 1: "🥈", 2: "🥉"}

        for i, (doc, score) in enumerate(zip(docs, scores)):
            medal      = medals.get(i, "📄")
            rank_class = {0: "rank-1", 1: "rank-2", 2: "rank-3"}.get(i, "rank-other")
            preview    = doc[:280] + "…" if len(doc) > 280 else doc

            st.markdown(f"""
            <div class="result-card {rank_class}">
                <div style="display:flex;align-items:center;gap:0.6rem;flex-wrap:wrap;">
                    <span style="font-size:1.3rem;">{medal}</span>
                    <strong>Rank #{i+1}</strong>
                    <span class="score-badge">TF-IDF Score {score:.4f}</span>
                </div>
                <div style="margin-top:0.5rem;opacity:0.9;font-size:0.88rem;line-height:1.55;">
                    {preview}
                </div>
            </div>""", unsafe_allow_html=True)

            with st.expander(f"Read full text — Paper #{i+1}"):
                st.write(doc)

        st.markdown("---")

        # ── Summary ──
        with st.spinner("Extracting key sentences from each document…"):
            summary_items, _ = summarize_query(query, top_k_docs=top_k_slider, sentences_per_doc=summary_n)

        st.subheader(f"📝 Extractive Highlights ({len(summary_items)} sentences total)")
        st.markdown(
            "💡 **Note on Summarization:** These are *extractive* highlights. The algorithm (TextRank) mathematically identifies and pulls out the most central sentences directly from the source text. It does not read and re-write the text like a Large Language Model (Abstractive Summarization) does."
        )

        if summary_items:
            for item in summary_items:
                medal = medals.get(item['rank'] - 1, f"📄")
                st.markdown(f"""
                <div class="summary-item">
                    <span class="summary-num">{medal} Rank #{item['rank']}</span> {item['sentence']}
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No highlights could be extracted for this query — try broadening the search terms.")

# ───────────────────────────────────────────────────────────────────────────
# SIDEBAR — About
# ───────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
This system performs **traditional NLP analysis** on ArXiv research papers.

**Pipeline:**
1. Text Preprocessing (spaCy)
2. TF-IDF Feature Extraction (trigrams, sublinear_tf)
3. LDA Topic Modeling (auto-optimised k)
4. TextRank Summarization
5. TF-IDF Retrieval
""")

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
- **Python**, spaCy, scikit-learn
- **Gensim** (LDA)
- **NetworkX** (TextRank PageRank)
- **Streamlit** (UI)
- **WordCloud**, Matplotlib
""")

    st.markdown("---")
    stats = data["eda_stats"]
    st.markdown(f"**📊 Corpus:** {stats['num_documents']:,} papers")
    st.markdown(f"**📖 Vocab:** {stats['vocabulary_size']:,} terms")
    st.markdown(f"**🏷️ Topics:** {data['coherence']['best_k']} (LDA optimised)")

