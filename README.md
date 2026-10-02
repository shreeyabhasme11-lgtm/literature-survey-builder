# Explainable Literature Survey Builder

A fully local, LLM-free web app that generates a literature survey from multiple
academic PDFs using extractive NLP, topic modeling, and coherence optimization.

## Setup

### 1. Install Tesseract (for OCR fallback)

**Linux/macOS:**
```bash
sudo apt install tesseract-ocr   # Debian/Ubuntu
brew install tesseract            # macOS
```

**Windows:**
Download from https://github.com/UB-Mannheim/tesseract/wiki  
Set env var: `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`

### 2. Python environment

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords
```

### 3. Run

```bash
streamlit run app/main.py
```

## Architecture
PDFs → Extract → Clean → Segment → TF-IDF → TextRank/LexRank
→ LDA Topics → Section Classify → Coherence Optimize
→ Template Compose → Explainability Trace → Export PDF/JSON


No LLM. No cloud. No API keys. Fully deterministic.

## Extending

- Add `sentence-transformers` for embedding-based ranking (set `USE_EMBEDDINGS=True` in config)
- Set `USE_BERTOPIC=True` in config for BERTopic instead of LDA
- Swap `VECTOR_STORE="chroma"` in config to use ChromaDB
- Adjust `TOP_K_PER_SECTION`, `REDUNDANCY_THRESHOLD` in `config.py`


literature-survey-builder/
├── requirements.txt
├── README.md
├── app/
│   ├── __init__.py
│   ├── config.py                        ← all tuneable constants
│   ├── main.py                          ← streamlit entrypoint
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── pages.py                     ← tab renderers
│   │   ├── components.py                ← reusable widgets
│   │   ├── state.py                     ← st.session_state helpers
│   │   └── styles.css
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── uploader.py
│   │   ├── validator.py
│   │   └── manifest.py
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py
│   │   ├── ocr_engine.py
│   │   ├── text_cleaner.py
│   │   └── sentence_splitter.py
│   ├── metadata/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   └── metadata_manager.py
│   ├── representation/
│   │   ├── __init__.py
│   │   ├── tfidf_vectorizer.py
│   │   ├── embedding_vectorizer.py
│   │   └── similarity.py
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── faiss_store.py
│   │   ├── chroma_store.py
│   │   └── store_factory.py
│   ├── ranking/
│   │   ├── __init__.py
│   │   ├── textrank_ranker.py
│   │   ├── lexrank_ranker.py
│   │   └── redundancy_filter.py
│   ├── topics/
│   │   ├── __init__.py
│   │   ├── bertopic_modeler.py
│   │   ├── lda_modeler.py
│   │   └── topic_labeler.py
│   ├── sections/
│   │   ├── __init__.py
│   │   ├── keyword_rules.py
│   │   ├── section_classifier.py
│   │   └── section_builder.py
│   ├── coherence/
│   │   ├── __init__.py
│   │   ├── subtopic_clusterer.py
│   │   ├── sentence_orderer.py
│   │   ├── transition_rules.py
│   │   ├── coherence_scorer.py
│   │   └── diversity_controller.py
│   ├── composition/
│   │   ├── __init__.py
│   │   ├── templates.py
│   │   ├── citation_formatter.py
│   │   └── survey_composer.py
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── evidence_trace.py
│   │   ├── score_explainer.py
│   │   ├── topic_explainer.py
│   │   └── ordering_explainer.py
│   ├── export/
│   │   ├── __init__.py
│   │   ├── html_renderer.py
│   │   ├── pdf_exporter.py
│   │   └── json_exporter.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── file_utils.py
│       └── text_utils.py
├── data/
│   ├── uploads/
│   ├── extracted/
│   ├── cleaned/
│   ├── sentences/
│   ├── vectors/
│   ├── topics/
│   ├── outputs/
│   └── traces/
└── tests/
    ├── test_extraction.py
    ├── test_cleaning.py
    ├── test_sentences.py
    ├── test_ranking.py
    ├── test_topics.py
    ├── test_sections.py
    ├── test_coherence.py
    └── test_export.py