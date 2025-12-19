import numpy as np
from scipy.special import expit
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from .config import RETRIEVER_MODEL, RERANKER_MODEL, EMBED_CACHE_DIR, TOP_K, QUERY_TOP_K, RERANK_TOP
from .data_manager import syn, doc_store

print("Initializing search engine...")
embedder = SentenceTransformer(RETRIEVER_MODEL)

# Synthetic queries embeddings
syn_cache = EMBED_CACHE_DIR / "syn_emb.npy"
syn_texts = syn["simple_question"].astype(str).tolist() if not syn.empty else []

if syn_cache.exists() and len(syn_texts) > 0:
    try:
        syn_emb = np.load(syn_cache)
        if syn_emb.shape[0] != len(syn_texts):
            print("Cached synthetic embeddings size mismatch. Recomputing...")
            raise ValueError("Size mismatch")
        print("Loaded synthetic embeddings from cache.")
    except Exception:
        print("Encoding synthetic queries...")
        syn_emb = embedder.encode(syn_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
        np.save(syn_cache, syn_emb)
else:
    if syn_texts:
        print("Encoding synthetic queries...")
        syn_emb = embedder.encode(syn_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
        np.save(syn_cache, syn_emb)
    else:
        syn_emb = np.array([])

# Index (FAISS or Sklearn)
USE_FAISS = False
faiss_index = None
nn = None

if len(syn_emb) > 0:
    try:
        import faiss
        USE_FAISS = True
        dim = syn_emb.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(syn_emb.astype(np.float32))
        print("Built FAISS index for synthetic queries.")
    except Exception:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(max(QUERY_TOP_K, TOP_K), len(syn_emb)), metric="cosine").fit(syn_emb)
        print("FAISS not available or failed — using sklearn NearestNeighbors for retrieval.")
else:
    print("Warning: No synthetic queries to index.")

# Reranker
reranker = CrossEncoder(RERANKER_MODEL)
print("Loaded cross-encoder reranker.")

# Chunk embeddings
chunk_ids = list(doc_store.keys())
chunk_texts = [doc_store[cid]["snippet_text"] for cid in chunk_ids]
chunk_emb_cache = EMBED_CACHE_DIR / "chunk_emb.npy"

if chunk_emb_cache.exists() and len(chunk_texts) > 0:
    try:
        chunk_embs = np.load(chunk_emb_cache)
        if chunk_embs.shape[0] != len(chunk_texts):
             raise ValueError("Size mismatch")
        print("Loaded chunk embeddings from cache.")
    except Exception:
        print("Encoding chunk embeddings for similarity signals...")
        chunk_embs = embedder.encode(chunk_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
        np.save(chunk_emb_cache, chunk_embs)
else:
    if chunk_texts:
        print("Encoding chunk embeddings for similarity signals...")
        chunk_embs = embedder.encode(chunk_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
        np.save(chunk_emb_cache, chunk_embs)
    else:
        chunk_embs = np.array([])

chunkid_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}

def retrieve_candidate_chunk_ids(query, top_k=TOP_K):
    if len(syn_emb) == 0:
        return []
    
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    if USE_FAISS and faiss_index:
        D, I = faiss_index.search(q_emb.astype(np.float32), top_k)
        indices = I[0].tolist()
    elif nn:
        n_neigh = min(top_k, len(syn_emb))
        dists, indices = nn.kneighbors(q_emb.reshape(1, -1), n_neighbors=n_neigh)
        indices = indices[0].tolist()
    else:
        return []
        
    policy_ids = syn.iloc[indices]["policy_id"].astype(str).tolist()
    seen = set(); uniq = []
    for pid in policy_ids:
        if pid not in seen:
            seen.add(pid); uniq.append(pid)
    return uniq

def rerank_chunks_with_probs(query, chunk_ids_to_rank, top_n=RERANK_TOP):
    if not chunk_ids_to_rank:
        return []

    pairs = []
    valid_ids = []
    for pid in chunk_ids_to_rank:
        info = doc_store.get(pid)
        if info:
            pairs.append([query, info["snippet_text"]])
            valid_ids.append(pid)

    if not pairs:
        return []

    logits = reranker.predict(pairs, show_progress_bar=False)
    logits = np.array(logits).squeeze()
    if logits.ndim > 1:
        logits = logits.reshape(-1)

    reranker_probs = expit(logits)
    
    # Handle single item case (scalar)
    if reranker_probs.ndim == 0:
        reranker_probs = np.array([reranker_probs])
        logits = np.array([logits])

    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    q_vec = q_emb[0] if hasattr(q_emb, 'shape') and len(q_emb.shape) > 1 else q_emb

    cos_sims = []
    for pid in valid_ids:
        idx = chunkid_to_idx.get(pid)
        if idx is None:
            cos_sims.append(0.0)
        else:
            cos = float(np.dot(q_vec, chunk_embs[idx]))
            cos_scaled = (cos + 1.0) / 2.0
            cos_sims.append(cos_scaled)

    alpha = 0.75
    combined_scores = alpha * reranker_probs + (1.0 - alpha) * np.array(cos_sims)

    results = []
    for i, (pid, prob, cosv, comb) in enumerate(zip(valid_ids, reranker_probs, cos_sims, combined_scores)):
        info = doc_store[pid]
        results.append({
            "policy_id": pid,
            "base_id": info.get("base_id"),
            "risk_category": info.get("risk_category"),
            "snippet_text": info.get("snippet_text"),
            "reranker_logit": float(logits[i]) if i < len(logits) else 0.0,
            "reranker_prob": float(prob) if i < len(reranker_probs) else 0.0,
            "cos_sim": float(cosv),
            "combined_score": float(comb)
        })

    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return results[:top_n]
