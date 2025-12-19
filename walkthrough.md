# Refactoring Walkthrough

I have successfully refactored the `risk_classifier_v7 (1).py` monolithic script into a structured Python package `risk_classifier` to improve maintainability and extensibility.

## Changes Created

### New Directory Structure
The project is now organized as follows:

- **`risk_classifier/`**: Root package directory.
    - **`config.py`**: Centralized configuration (paths, model names, thresholds).
    - **`utils.py`**: Shared utilities like PII detection and safe CSV reading.
    - **`data_manager.py`**: Responsible for loading the CSV data (`synthetic_queries`, `chunks`) and creating the document store.
    - **`toxicity.py`**: Encapsulates the toxicity detection model (Toxic-BERT) and lexicon logic.
    - **`pdf_processor.py`**: Handles PDF text extraction and OCR (using PyMuPDF and Tesseract).
    - **`search_engine.py`**: Manages the embedding model (`all-mpnet-base-v2`), the reranker (`ms-marco-MiniLM`), and the search index (FAISS/Sklearn).
    - **`risk_assessment.py`**: Contains pure logic for scoring and risk aggregation.
    - **`pipeline.py`**: Orchestrates the components to provide `classify_pdf` and `match_query` functions.
    - **`main.py`**: The entry point for running the application from the CLI.

### Files Added
- **`requirements.txt`**: Lists all necessary Python libraries.
- **`README.md`**: Instructions for installation and usage.

## Verification
I attempted to verify the installation but found that the current environment lacks the necessary dependencies (e.g., `transformers`, `pandas`). You will need to install them using:
```bash
pip install -r requirements.txt
```

After installation, you can run the classifier with:
```bash
python -m risk_classifier.main
```
