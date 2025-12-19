from transformers import pipeline
import re
import unicodedata

print("Loading toxicity classifier (may take a moment)...")
try:
    # some HF versions: top_k=None returns list/dicts correctly
    toxicity_clf = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)
except Exception as e:
    print(f"Warning: Could not load toxicity model. Error: {e}")
    toxicity_clf = None

TOXIC_LEXICON = [
    "fuck","die","kill","bomb","terror","i hate","immigrant","immigrants",
    "nigger","bitch","slur","go die","go to hell","fascist","kill yourself"
]

def normalize_text_for_lexicon(t: str):
    t = str(t).lower()
    t = re.sub(r'(.)\1{2,}', r'\1\1', t)
    t = t.replace('4','a').replace('3','e').replace('1','i').replace('0','o').replace('5','s')
    t = unicodedata.normalize('NFKD', t)
    return t

def lexicon_hits(text: str):
    t = normalize_text_for_lexicon(text)
    return [w for w in TOXIC_LEXICON if w in t]

def detect_toxicity_spans(text: str, sentence_split_regex=r'(?<=[.!?\n])\s+'):
    if not text or not text.strip():
        return [], {"notice":"green","message":"No toxicity/hate/threat detected with current detectors."}
    sentences = [s.strip() for s in re.split(sentence_split_regex, text) if s.strip()]
    spans = []
    for i, s in enumerate(sentences):
        lx = lexicon_hits(s)
        try:
            out = toxicity_clf(s[:1000]) if toxicity_clf else []
        except Exception:
            out = []
        per_label = {}
        # Several HF pipeline shapes - handle robustly
        if out and isinstance(out, list) and isinstance(out[0], dict) and 'label' in out[0]:
            # list of dicts form
            for d in out:
                per_label[d['label'].lower()] = float(d.get('score', 0.0))
        elif out and isinstance(out[0], list):
            # nested list form
            for d in out[0]:
                per_label[d['label'].lower()] = float(d.get('score', 0.0))
        else:
            # fallback: attempt to parse
            try:
                for d in out:
                    if isinstance(d, dict) and 'label' in d:
                        per_label[d['label'].lower()] = float(d.get('score', 0.0))
            except Exception:
                per_label = {}

        toxic_labels = ['toxic','severe_toxicity','threat','insult','identity_hate','obscene']
        ml_score = max([per_label.get(lbl, 0.0) for lbl in toxic_labels]) if per_label else 0.0

        categories = []
        if per_label.get('threat', 0.0) > 0.35 or re.search(r'\b(kill|bomb|die|harm|destroy)\b', s, re.I):
            categories.append('Threat')
        if per_label.get('identity_hate', 0.0) > 0.25 or any(w in ' '.join(lx) for w in ['immigrant','nigger','fascist']):
            categories.append('Hate')
        if per_label.get('insult', 0.0) > 0.25 or per_label.get('obscene', 0.0) > 0.25 or lx:
            categories.append('Toxic/Profanity')

        spans.append({
            'idx': i,
            'text': s,
            'lex_hits': lx,
            'ml_score': ml_score,
            'per_label': per_label,
            'categories': list(set(categories))
        })

    all_cats = set(c for sp in spans for c in sp['categories'])
    doc_toxic_score = max([sp['ml_score'] for sp in spans]) if spans else 0.0
    if not all_cats:
        summary = {"notice":"green","message":"No toxicity/hate/threat detected with current detectors.","doc_toxic_score": doc_toxic_score}
    else:
        summary = {"notice":"red","message": f"Detected categories: {', '.join(sorted(all_cats))}", "categories": sorted(all_cats), "doc_toxic_score": doc_toxic_score}
    return spans, summary
