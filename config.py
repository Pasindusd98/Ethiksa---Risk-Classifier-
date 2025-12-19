import os
from pathlib import Path

# Files
SYN_CSV = "synthetic_queries_v3.csv"
AI_CHUNKS_CSV = "ai_act_chunks.csv"
GDPR_CHUNKS_CSV = "gdpr_chunks.csv"

# Models & index settings
RETRIEVER_MODEL = "all-mpnet-base-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_CACHE_DIR = Path("emb_cache")
EMBED_CACHE_DIR.mkdir(exist_ok=True)

# Retrieval / rerank params
TOP_K = 10           # used for PDF page retrieval (keeps small)
QUERY_TOP_K = 50     # larger for interactive queries to increase recall
RERANK_TOP = 6
SIM_THRESHOLD = 0.60

# PDF / OCR
OCR_ZOOM = 2.0
OCR_LANG = "eng"
OCR_CONF_THRESHOLD = 30.0

# Risk thresholds
RISK_LEVEL_THRESHOLDS = {"high": 0.7, "medium": 0.4}

OUT_DIR = Path("rc_outputs")
OUT_DIR.mkdir(exist_ok=True)
