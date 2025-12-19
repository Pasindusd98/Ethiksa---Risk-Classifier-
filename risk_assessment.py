from .config import RISK_LEVEL_THRESHOLDS

def score_to_severity(score: float):
    if score >= RISK_LEVEL_THRESHOLDS["high"]:
        return "High"
    if score >= RISK_LEVEL_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"

def aggregate_document_risk(violations_list, safety_doc_score):
    if not violations_list:
        if safety_doc_score >= RISK_LEVEL_THRESHOLDS["high"]:
            return "High"
        if safety_doc_score >= RISK_LEVEL_THRESHOLDS["medium"]:
            return "Medium"
        return "Low"

    severities = [v.get("violation_severity","Low") for v in violations_list]
    counts = {"High":0,"Medium":0,"Low":0}
    for s in severities:
        counts[s] = counts.get(s, 0) + 1

    if counts["High"] > 0:
        return "High"
    total = counts["High"] + counts["Medium"] + counts["Low"]
    if counts["Medium"] > total/2:
        return "Medium"
    if safety_doc_score >= RISK_LEVEL_THRESHOLDS["high"]:
        return "High"
    if safety_doc_score >= RISK_LEVEL_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"
