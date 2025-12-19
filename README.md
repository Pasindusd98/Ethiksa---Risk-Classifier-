# Risk Classifier Project

This project classifies valid PDF documents and text queries based on AI Act and GDPR policies.
It has been refactored from a single script into a modular Python package.

## Project Structure

- `risk_classifier/`: The Python package.
  - `config.py`: Configuration constants (Paths, Thresholds, Model names).
  - `utils.py`: Helper functions (PII detection, CSV reading).
  - `data_manager.py`: Loads and prepares the `synthetic_queries` and `chunks` data.
  - `search_engine.py`: Handles SentenceTransformer embeddings, FAISS/Sklearn indexing, and retrieval.
  - `toxicity.py`: Toxicity detection using HuggingFace pipelines.
  - `pdf_processor.py`: PDF extraction and OCR fallback (requires PyMuPDF, Tesseract).
  - `risk_assessment.py`: Scoring logic and risk aggregation rules.
  - `pipeline.py`: Main logic flows `classify_pdf` and `match_query`.
  - `main.py`: CLI entry point.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **System Requirements**: 
   - Install `tesseract-ocr` for OCR functionality if needed.

## Usage

### Run from Command Line
To run the classifier on the most recent PDF in the directory (or a specific file):

```bash
python -m risk_classifier.main
```

### Use in Python Code
```python
from risk_classifier.pipeline import classify_pdf, match_query

# Classify a PDF
result = classify_pdf("my_document.pdf")
print(result["risk_level"])

# Check a text query
query_result = match_query("How do I scrape user data?")
print(query_result["decision"])
```

## Data
The system expects `synthetic_queries_v3.csv`, `ai_act_chunks.csv`, and `gdpr_chunks.csv` in the working directory (or paths configured in `config.py`).
