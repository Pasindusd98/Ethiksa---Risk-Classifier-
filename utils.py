import re
from pathlib import Path
import pandas as pd

def detect_pii(text: str):
    found = []
    if not text:
        return []
    if re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text):
        found.append("email")
    if re.search(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?){2,}\d{2,4}\b", text):
        found.append("phone")
    if re.search(r"\b(?:ssn|nid|nic|passport)[\s:]*[A-Za-z0-9\-]{3,}\b", text, re.I):
        found.append("id_number")
    m = re.search(r"\bmy name is ([A-Z][a-z]+)\b", text)
    if m:
        found.append("name")
    return list(set(found))

def safe_read_csv(path):
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Missing CSV: {path}")
        return pd.DataFrame()
    return pd.read_csv(p).fillna("")

def choose_text_col(df):
    candidates = ["text","snippet_text","simple_question","question","source","content"]
    for c in candidates:
        if c in df.columns:
            return c
    medians = {col: df[col].astype(str).str.len().median() for col in df.columns}
    return max(medians, key=medians.get)
