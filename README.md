# Research Topic Analysis & Summarization System

### Traditional NLP-Based Research Paper Analyzer

---

## Problem Statement

The rapid growth of academic research publications makes it difficult for researchers and students to efficiently understand developments within a specific domain. Manually reading multiple research papers to extract key insights is time-consuming and inefficient.

This project addresses the challenge of automatically **retrieving, analyzing, and summarizing research papers** related to a user-defined topic using **traditional Natural Language Processing (NLP)** techniques.

The developed system acts as an **intelligent research assistant** that:

* Retrieves relevant academic papers
* Identifies dominant research themes
* Generates concise extractive summaries
* Presents insights through an interactive web interface

The project strictly avoids Large Language Models (LLMs) and instead relies on explainable statistical NLP methods.

---

## Data Description

### Data Source

Research papers are dynamically collected using the **arXiv API**, ensuring access to real-world academic literature across multiple domains:

* Artificial Intelligence (cs.AI)
* Physics (physics.gen-ph)
* Mathematics (math.CO)
* Economics (econ.GN)
* Quantitative Biology (q-bio.BM)

---

### Dataset Features

Each document consists of:

* Research Paper Title
* Research Paper Abstract

The final document representation used in analysis is:

```
Title + Abstract
```

Abstracts are used instead of full PDFs because they contain dense research information while significantly reducing computational complexity.

---

### Stored Data Artifacts

```
data/
 ├── raw_docs.pkl
 ├── clean_docs.pkl
 ├── tfidf_vectorizer.pkl
 ├── tfidf_matrix.pkl
 └── lda_model.pkl
```

---

## Exploratory Data Analysis (EDA)

Exploratory analysis was conducted to understand corpus characteristics:

* Vocabulary distribution across domains
* Average tokens per research paper
* Frequent domain-specific keywords
* Topic diversity within collected papers

### Key Observations

* Research titles strongly influence topic similarity.
* Abstracts provide sufficient semantic information.
* Common academic terms are automatically suppressed using TF-IDF weighting.
* Domain keywords dominate topic formation.

These insights validated the use of statistical NLP techniques for analysis.

---

## Methodology

The system follows a structured end-to-end NLP pipeline.

---

### 1️⃣ Data Ingestion

* Papers fetched using arXiv API
* Multi-domain corpus construction
* Duplicate paper removal
* Automated dataset generation

---

### 2️⃣ Text Preprocessing

Performed using **spaCy NLP pipeline**:

* Lowercase normalization
* Stopword removal
* Lemmatization
* Punctuation removal
* Citation & LaTeX noise removal
* POS filtering (NOUN, PROPN, ADJ)

Purpose:
Improve semantic quality and remove linguistic noise.

---

### 3️⃣ TF-IDF Vectorization

TF-IDF converts documents into numerical vectors based on:

* **Term Frequency (TF)** – importance within a document
* **Inverse Document Frequency (IDF)** – rarity across corpus

TF-IDF is used twice:

1. Document similarity for query-based retrieval
2. Sentence similarity for summarization

Reason:
Provides interpretable statistical representation without deep learning models.

---

### 4️⃣ Query-Based Document Retrieval

When a user enters a research topic:

1. Query is transformed using trained TF-IDF vectorizer
2. Cosine similarity is computed
3. Top-K relevant papers are selected

This ensures focused topic analysis instead of processing the entire dataset.

---

### 5️⃣ Topic Modeling — LDA

Latent Dirichlet Allocation (LDA) identifies hidden thematic structures within research papers.

Output:

* Topic clusters
* Dominant keywords per topic

Purpose:
Reveal conceptual research trends automatically.

---

### 6️⃣ Extractive Summarization — TextRank

TextRank is a graph-based ranking algorithm inspired by Google PageRank.

#### Process:

1. Retrieved documents are combined
2. Text is split into sentences
3. Sentences converted into TF-IDF vectors
4. Sentence similarity matrix computed
5. Similarity graph constructed
6. PageRank algorithm applied
7. Highest-ranked sentences selected
8. Sentences reordered for readability

Final output:
A coherent extractive summary representing core research insights.

---

## Evaluation Strategy

Since this project performs **analysis and summarization rather than prediction**, traditional metrics such as Accuracy, F1-score, MAE, or RMSE are not applicable.

Evaluation is performed using:

* Retrieval relevance inspection
* Topic interpretability analysis
* Summary coherence validation
* Runtime efficiency assessment

Results demonstrate effective topic discovery and meaningful summary generation.

---

## Optimization Techniques

Performance improvements include:

* Abstract-only processing
* Dataset size control
* Duplicate removal
* spaCy batch processing (`nlp.pipe`)
* Precomputed TF-IDF matrices
* Streamlit caching mechanisms

These optimizations improve speed, scalability, and deployment stability.

---

## System Architecture

```
arXiv API
     ↓
Data Preprocessing
     ↓
TF-IDF Vectorization
     ↓
User Query Input
     ↓
Cosine Similarity Retrieval
     ↓
Relevant Documents
     ↓
LDA Topic Modeling
     ↓
TextRank Summarization
     ↓
Streamlit Web Interface
```

---

## Deployment

The project is deployed using:

✅ Streamlit Cloud / Hugging Face Spaces
(Add live deployment URL here)

---

## Team Contribution

| Team Member | Contribution                    |
| ----------- | ------------------------------- |
| Akhilesh Kumar   | Data ingestion & preprocessing  |
| Dhruv Ramani   | TF-IDF retrieval implementation |
| Lakshya Agarwal  | LDA topic modeling              |
| Kavya Jain   | TextRank summarization & UI     |

---

## Technologies Used

* Python
* spaCy
* Scikit-learn
* Gensim
* NetworkX
* Streamlit
* arXiv API

---

## Future Scope

The system architecture is designed for extension into:

* Autonomous research agents
* Real-time academic search
* Multi-document reasoning
* Research question answering systems

---

## Conclusion

This project demonstrates how traditional NLP techniques can effectively analyze academic literature without relying on large language models.

By integrating:

* TF-IDF for document representation
* LDA for topic discovery
* TextRank for extractive summarization

the system provides an interpretable and scalable solution for automated research understanding.

---
