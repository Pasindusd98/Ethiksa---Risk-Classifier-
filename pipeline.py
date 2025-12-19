import time
import json
from .config import TOP_K, RERANK_TOP, SIM_THRESHOLD, RISK_LEVEL_THRESHOLDS, QUERY_TOP_K, OUT_DIR
from .utils import detect_pii
from .toxicity import detect_toxicity_spans
from .pdf_processor import extract_text_from_pdf
from .search_engine import retrieve_candidate_chunk_ids, rerank_chunks_with_probs, chunk_ids
from .risk_assessment import score_to_severity, aggregate_document_risk

def get_violated_act_name(base_id):
    if not base_id:
        return None
    if base_id.startswith("EU_AI_Act"):
        return "EU AI Act"
    if base_id.startswith("GDPR"):
        return "GDPR"
    return base_id

def classify_pdf(pdf_path, run_per_page=True, top_k=TOP_K):
    start_time = time.time()
    # Ensure chunk_ids is available even if search_engine failed slightly or is empty
    current_chunk_ids = chunk_ids if chunk_ids else []

    pages = extract_text_from_pdf(pdf_path)
    full_text = "\n\n".join([p['text'] for p in pages if p['text']])

    pii_doc = detect_pii(full_text)
    spans_doc, safety_summary_doc = detect_toxicity_spans(full_text)
    doc_toxic_score = float(safety_summary_doc.get("doc_toxic_score", 0.0) if isinstance(safety_summary_doc, dict) else 0.0)

    candidate_ids_doc = retrieve_candidate_chunk_ids(full_text, top_k=top_k)
    # if no candidate ids (rare), use all chunk_ids
    if not candidate_ids_doc and current_chunk_ids:
        candidate_ids_doc = current_chunk_ids.copy()

    reranked_doc = rerank_chunks_with_probs(full_text, candidate_ids_doc, top_n=RERANK_TOP)

    violations = {}
    def record_match(match, page_num=None, context_text=None):
        pid = match.get("policy_id")
        if not pid:
            return
        cur = violations.get(pid)
        score = float(match.get("combined_score", 0.0))
        if cur is None:
            violations[pid] = {
                "policy_id": pid,
                "base_id": match.get("base_id"),
                "risk_category": match.get("risk_category"),
                "best_score": score,
                "occurrences": 1,
                "pages": [page_num] if page_num is not None else [],
                "contexts": [context_text] if context_text else [match.get("snippet_text","")]
            }
        else:
            cur["occurrences"] += 1
            if score > cur["best_score"]:
                cur["best_score"] = score
            if page_num is not None and page_num not in cur["pages"]:
                cur["pages"].append(page_num)
            if context_text:
                cur["contexts"].append(context_text)

    for m in reranked_doc:
        record_match(m, page_num=None, context_text=m.get("snippet_text"))

    page_evidence = []
    if run_per_page:
        for p in pages:
            text = p.get('text','').strip()
            if not text:
                page_evidence.append({
                    "page_num": p['page_num'],
                    "is_selectable": p['is_selectable'],
                    "pii": [],
                    "safety_summary": {"notice":"green","message":"No text"},
                    "top_matches": []
                })
                continue
            pii_page = detect_pii(text)
            spans_page, safety_summary_page = detect_toxicity_spans(text)
            cand = retrieve_candidate_chunk_ids(text, top_k=top_k)
            if not cand and current_chunk_ids:
                cand = current_chunk_ids.copy()
            reranked_page = rerank_chunks_with_probs(text, cand, top_n=RERANK_TOP)
            for m in reranked_page:
                record_match(m, page_num=p['page_num'], context_text=text[:400])
            page_evidence.append({
                "page_num": p['page_num'],
                "is_selectable": p['is_selectable'],
                "pii": pii_page,
                "safety_summary": safety_summary_page,
                "top_matches": reranked_page,
                "ocr_boxes": p.get('ocr_boxes')
            })

    violations_list = []
    for pid, info in violations.items():
        violations_list.append({
            "policy_id": pid,
            "base_id": info.get("base_id"),
            "risk_category": info.get("risk_category"),
            "best_score": float(info.get("best_score",0.0)),
            "occurrences": info.get("occurrences",0),
            "pages": info.get("pages",[]),
            "contexts": info.get("contexts",[])
        })
    violations_list.sort(key=lambda x: x["best_score"], reverse=True)
    for v in violations_list:
        v["violation_severity"] = score_to_severity(v["best_score"])

    highest_policy_score = violations_list[0]["best_score"] if violations_list else 0.0
    overall_confidence = max(highest_policy_score, doc_toxic_score)
    overall_risk_level = aggregate_document_risk(violations_list, doc_toxic_score)
    violations_above_threshold = [v for v in violations_list if v["best_score"] >= SIM_THRESHOLD]

    counts = {"High":0,"Medium":0,"Low":0}
    for v in violations_list:
        counts[v.get("violation_severity","Low")] += 1
    guideline_lines = []
    guideline_lines.append(f"Detected {len(violations_list)} potential policy violations: {counts['High']} High, {counts['Medium']} Medium, {counts['Low']} Low.")
    guideline_lines.append(f"Document toxicity score: {doc_toxic_score:.2f}.")
    guideline_lines.append("Aggregation rule: if any violation is High -> document is High risk; else if majority of violations are Medium -> Medium risk; else Low risk.")
    guideline_lines.append(f"Overall decision: {overall_risk_level} (confidence {overall_confidence:.2f}).")
    sample_examples = []
    for v in violations_list[:3]:
        sample_examples.append(f"{v['policy_id']} ({v['violation_severity']}, score {v['best_score']:.2f}) - pages {v['pages']}")
    if sample_examples:
        guideline_lines.append("Examples: " + " | ".join(sample_examples))
    guideline_text = " ".join(guideline_lines)

    out = {
        "source": str(pdf_path),
        "num_pages": len(pages),
        "violations_all": violations_list,
        "violations_above_threshold": violations_above_threshold,
        "num_violations": len(violations_list),
        "num_violations_above_threshold": len(violations_above_threshold),
        "overall_confidence": overall_confidence,
        "risk_level": overall_risk_level,
        "pii_detected_doc": pii_doc,
        "safety_summary_doc": safety_summary_doc,
        "page_evidence": page_evidence,
        "guideline": {
            "text": guideline_text,
            "counts": counts,
            "aggregation_rule": "any-High -> High; else majority-Medium -> Medium; else Low",
            "examples": sample_examples
        },
        "duration_s": time.time() - start_time
    }

    ts = int(time.time()*1000)
    out_fname = OUT_DIR / f"pdf_match_{ts}.json"
    with open(out_fname, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Saved PDF match ->", out_fname)
    return out

def match_query(query: str, query_top_k=QUERY_TOP_K):
    start = time.time()
    query = str(query).strip()
    if not query:
        return {"query": query, "decision": "no_input"}

    pii_detected = detect_pii(query)
    spans, safety_summary = detect_toxicity_spans(query)
    doc_toxic_score = float(safety_summary.get("doc_toxic_score", 0.0) if isinstance(safety_summary, dict) else 0.0)

    candidate_ids = retrieve_candidate_chunk_ids(query, top_k=query_top_k)
    # fallback: if no candidate ids, use all chunk ids
    if not candidate_ids and chunk_ids:
        candidate_ids = chunk_ids.copy()

    reranked_results = rerank_chunks_with_probs(query, candidate_ids, top_n=RERANK_TOP)

    result = {
        "query": query,
        "violated_act": None,
        "policy_id": None,
        "risk_category": "Low",
        "confidence": 0.0,
        "pii_detected": pii_detected,
        "safety_summary": safety_summary,
        "reason": "No direct policy violation detected.",
        "duration_s": time.time() - start,
        "decision": "Low"
    }

    if reranked_results:
        top_match = reranked_results[0]
        conf = float(top_match.get("combined_score", 0.0))
        result["confidence"] = conf
        result["policy_id"] = top_match.get("policy_id")
        result["violated_act"] = get_violated_act_name(top_match.get("base_id"))
        result["risk_category"] = top_match.get("risk_category")
        result["reason"] = top_match.get("snippet_text")
        # decision based on thresholds
        if conf >= SIM_THRESHOLD:
            result["decision"] = score_to_severity(conf)
        else:
            # if doc toxicity is higher, escalate
            result["decision"] = score_to_severity(max(conf, doc_toxic_score))
            # keep confidence as max of both
            result["confidence"] = max(conf, doc_toxic_score)

    # finalize: if toxicity alone is high and there was no policy match, escalate
    if (result["confidence"] < RISK_LEVEL_THRESHOLDS["high"]) and doc_toxic_score >= RISK_LEVEL_THRESHOLDS["high"]:
        result["decision"] = "High"
        result["confidence"] = max(result["confidence"], doc_toxic_score)
    elif (result["confidence"] < RISK_LEVEL_THRESHOLDS["medium"]) and doc_toxic_score >= RISK_LEVEL_THRESHOLDS["medium"]:
        if result["decision"] == "Low":
            result["decision"] = "Medium"
            result["confidence"] = max(result["confidence"], doc_toxic_score)

    result["duration_s"] = time.time() - start
    return result
