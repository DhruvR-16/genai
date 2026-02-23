# 📚 Intelligent Research Topic Analysis System

An NLP-powered research analysis system that dynamically processes **100 recent ArXiv research papers** across diverse scientific disciplines (AI, Physics, Biology, Economics, Math). It extracts topics, keywords, and generates extractive summaries using traditional machine learning techniques.

---

## 🏗️ System Architecture

The project has been simplified into a flattened, focused structure:

```text
genai_copy/
├── app/
│   └── streamlit_app.py      # Interactive Web UI (4 main tabs)
├── src/
│   ├── preprocess.py         # Text cleaning, LaTeX removal, tokenization
│   ├── tfidf.py              # TF-IDF feature vectorization
│   ├── topic_model.py        # LDA topic modeling + coherence optimization
│   ├── textrank_summary.py   # TextRank extractive summarization
│   ├── keyword_extractor.py  # Global and per-topic keyword extraction
│   ├── query_summarizer.py   # Query-based search + summary summarization
│   ├── search.py             # TF-IDF cosine similarity search engine
│   └── eda_stats.py          # Corpus exploratory data analysis logic
├── data/
│   └── processed/            # Generated .pkl data files
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🚀 Running the Pipeline

Make sure you have your virtual environment activated and dependencies installed:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Run the pipeline scripts in the following order:

```bash
# 1. Fetch 100 ArXiv papers and preprocess them
python src/preprocess.py

# 2. Build TF-IDF features
python src/tfidf.py

# 3. Train the LDA topic model (automatically optimizes topic count)
python src/topic_model.py

# 4. Generate extractive summaries using TextRank
python src/textrank_summary.py

# 5. Extract global and per-topic keywords
python src/keyword_extractor.py

# 6. Compute statistics for EDA UI
python src/eda_stats.py
```

---

## 🖥️ Launching the UI

```bash
streamlit run app/streamlit_app.py
```

The app will open at `http://localhost:8501` featuring:

1. **🔍 Search & Summary** — Query-based document retrieval + extractive natural sentence summary
2. **📊 EDA & Preprocessing** — Corpus statistics, word cloud, token frequency
3. **🧠 Topic Modeling** — LDA topics, coherence metrics, top keywords
4. **🔑 Keyword Extraction** — TF-IDF keywords, per-topic keywords

---

## 📁 Dataset

**Universal ArXiv Dataset**

- Source: ArXiv API (via `arxiv` python package)
- Papers dynamically fetched: 100 total
- Disciplines covered: `cs.AI` (Artificial Intelligence), `physics.gen-ph` (Physics), `q-bio.BM` (Biomolecules), `econ.GN` (Economics), `math.CO` (Combinatorics).
