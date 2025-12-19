import pandas as pd
import sys
from .config import SYN_CSV, AI_CHUNKS_CSV, GDPR_CHUNKS_CSV
from .utils import safe_read_csv, choose_text_col

print("Loading CSVs...")
syn = safe_read_csv(SYN_CSV)
ai_chunks = safe_read_csv(AI_CHUNKS_CSV)
gdpr_chunks = safe_read_csv(GDPR_CHUNKS_CSV)

if syn.empty or (ai_chunks.empty and gdpr_chunks.empty):
    print("One or more required CSVs are missing or empty. Place files and re-run.")
    # We won't exit here to allow importing for inspection, but functionality will be broken.

syn_text_col = choose_text_col(syn) if not syn.empty else None
if not syn.empty and syn_text_col:
    syn = syn.rename(columns={syn_text_col: "simple_question"})
    if "policy_id" not in syn.columns:
        print("[ERROR] synthetic CSV must have a 'policy_id' column mapping to chunk ids.")

chunks = pd.concat([ai_chunks, gdpr_chunks], ignore_index=True).fillna("")
chunk_text_col = choose_text_col(chunks) if not chunks.empty else None
if not chunks.empty and chunk_text_col:
    chunks = chunks.rename(columns={chunk_text_col: "snippet_text"})
    if "policy_id" not in chunks.columns:
         print("[ERROR] chunk CSVs must contain 'policy_id' column.")
    if "risk_category" not in chunks.columns:
        chunks["risk_category"] = chunks.get("risk_category", "")

doc_store = {}
if not chunks.empty:
    for _, r in chunks.iterrows():
        pid = str(r.get("policy_id", ""))
        if not pid: continue
        doc_store[pid] = {
            "snippet_text": str(r.get("snippet_text", "")),
            "risk_category": str(r.get("risk_category","")),
            "base_id": "_".join(pid.split("_")[:4]) if "_" in pid else pid
        }

print(f"Loaded {len(syn)} synthetic queries and {len(doc_store)} chunks.")
