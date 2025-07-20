import json
import pandas as pd
import numpy as np  
from typing import List, Dict
from difflib import SequenceMatcher
import re
import os
import time
from nltk.tokenize import word_tokenize

from retrieval_utils import RetrievalSystem  # Use absolute import if retrieval_system.py is in the same directory


def normalize_text(text: str) -> str:
    """
    Normalizza una stringa per confronti fuzzy:
      1) Sostituisce NBSP con spazio normale
      2) Rimuove caratteri di controllo (se ce ne sono)
      3) Comprime sequenze di spazi/tab in un singolo spazio
      4) Strip iniziale/finale
    """
    if not isinstance(text, str):
        return text

    # 1) NBSP → spazio
    txt = text.replace('\xa0', ' ')
    # 2) Rimuovi eventuali caratteri di controllo
    #    (qui potresti fare txt = ''.join(ch for ch in txt if ch.isprintable()) )
    # 3) Comprime multi‑spazio/tab/newline in uno spazio
    txt = ' '.join(txt.split())
    # 4) strip
    return txt.strip()



def fuzzy_match(span: str, text: str, threshold: float = 0.7) -> bool:
    """
    Returns True if a fuzzy string match between span and text is above the threshold.
    Uses SequenceMatcher ratio from difflib (can be replaced with rapidfuzz for better perf).
    
    The fuzzy_match function specifically implements this concept using 
    difflib.SequenceMatcher. This function calculates a "ratio" of similarity 
    between two strings. 
    If this ratio meets or exceeds a predefined threshold (defaulting to 0.7), 
    the strings are considered a fuzzy match.
    
    """
    span = span.lower()
    text = text.lower()
    span_norm = normalize_text(span)
    chunk_text_norm = normalize_text(text)
    return SequenceMatcher(None, span_norm, chunk_text_norm).ratio() >= threshold


def reference_label_match(span: str, text: str) -> bool:
    """
    Special case matcher for legal references like 'Article 5' or 'Recital 14'.
    We check if span appears as a phrase inside the retrieved text.
    """
    return span.lower() in text.lower()


def relaxed_span_match(span: str, chunk_text: str) -> bool:
    """
    Combined matching strategy:
    1. Exact match
    2. Reference phrase match
    3. Fuzzy semantic match
    """
    chunk_text = chunk_text.replace('\xa0', ' ')
    chunk_text = re.sub(r'\s+', ' ', chunk_text)
    chunk_text = chunk_text.strip()

    exact_match = span in chunk_text
    reference_label = reference_label_match(span, chunk_text)
    fuzzy_match_result = fuzzy_match(span, chunk_text)

    if exact_match:
        print(f"Exact match found")
        return True
    if reference_label:
        print(f"Reference label match found")
        return True
    if fuzzy_match_result:
        print(f"Fuzzy match found")
        return True
    
    return False


def get_match_across_subspans(gold_spans, top_k_chunks):
    """
    Per ogni sottospan in gold_spans, controlla se esiste almeno un chunk in top_k_chunks
    che soddisfi relaxed_span_match(sub_span, chunk["text"]). Ritorna True solo se *tutti*
    i sottospan sono stati trovati in almeno un chunk.
    """
    for sub_span in gold_spans:
        # Controllo se esiste almeno un chunk che matcha questo sub_span
        if not any(relaxed_span_match(sub_span, chunk["text"]) for chunk in top_k_chunks):
            print("-"*80)
            print("!!! Completely Failed to match sub_span:", sub_span)  
            print(f"in chunks: ")
            for chunk in top_k_chunks:
                print(chunk["text"])
                print()
            print("-"*80)
            return False
    
    
    return True


def evaluate_recall_at_k_relaxed(
    retriever,  # your RetrievalSystem
    qa_df: pd.DataFrame,
    k: int = 5,
) -> Dict:
    """
    Evaluate Recall@K using relaxed span matching (fuzzy, reference, or exact).
    """
    total = len(qa_df)
    recall = {"vector": 0, "bm25": 0, "hybrid": 0}

    for _, row in qa_df.iterrows():
        question = row["question"]
        gold_spans  = row["answer_span"] # now it's a list

        for mode in ["vector", "bm25", "hybrid"]:
            # Use getattr to access the appropriate retrieval function
            retrieval_fn = getattr(retriever, f"retrieve_{mode}")
            top_k_chunks = retrieval_fn(question, k)

            # Only count as a hit if *all* sub‐spans were found
            if get_match_across_subspans(gold_spans, top_k_chunks):
                recall[mode] += 1

    # Normalize counts into recall scores
    for mode in recall:
        recall[mode] = recall[mode] / total

    return recall



# 2) Function to log misses and do a loose span check
def inspect_missed_queries(rsys, qa_df, k=5, relaxed=True, method="vector"):
    """
    For each question where retrieve_vector fails to find the answer_span
    in the top-k, print:
      - The question
      - The true answer_span
      - The top-k retrieved chunk texts
      - Whether the answer_span appears anywhere in the entire chunk collection
    """
    # Preload all chunks’ texts for the loose check
    all_texts = [entry["text"] for entry in rsys.meta]

    print(f"Inspecting top-{k} retrievals for {len(qa_df)} questions...")

    # Iterate through each question-answer pair in the DataFrame
    for idx, row in qa_df.iterrows():
        question   = row["question"]
        gold_spans = row["answer_span"]

        if method == "vector":
            print(f"\nRetrieving top-{k} chunks for question #{idx} using vector retrieval...")
            # Retrieve top-k by vector
            results = rsys.retrieve_vector(question, k=k)
        elif method == "bm25":  
            print(f"\nRetrieving top-{k} chunks for question #{idx} using BM25 retrieval...")
            # Retrieve top-k by BM25
            results = rsys.retrieve_bm25(question, k=k) 
        elif method == "hybrid":
            print(f"\nRetrieving top-{k} chunks for question #{idx} using hybrid retrieval...")
            # Retrieve top-k by hybrid method
            results = rsys.retrieve_hybrid(question, k=k)
        
        # Check if we hit the span in the top-k
        if relaxed:
            # Use relaxed matching logic
            print(f"Using relaxed matching for question #{idx}...")
            hit = get_match_across_subspans(gold_spans, results)
        else:
            # Use strict containment check
            # This is the original logic: span must be a substring of the chunk text    
            hit = any(span in hit["text"] for hit in results)

        if not hit:
            print("\n" + "="*80)
            print(f"❌ MISS: Question #{idx}: {question!r}")
            print(f"   → Gold spans: {gold_spans!r}")
            print(f"   Top-{k} retrieved chunks:")
            for rank, chunk in enumerate(results, start=1):
                # Print first 100 chars of each chunk for brevity
                snippet = chunk["text"].replace("\n", " ")
                print(f"     {rank}. {snippet!r} ")

            # Loose check: does the span appear in *any* chunk?
            for span in gold_spans:
                print(f"   Checking span: {span!r}")
                # Check if the span appears in any of the preloaded texts
                found_anywhere = any(span in text for text in all_texts)
            
                if found_anywhere:
                    print("   ⚠️ Note: span exists somewhere in the corpus — maybe chunk boundaries split it.")
                else:
                    print("   ❌ Span not found anywhere — consider adjusting answer_span or source labels.")
        else:
            print(f"✅ HIT: Question #{idx}: {question!r} → Found span in top-{k} results.")
            print(f"   Gold span: {gold_spans!r}")
            # Optionally, you could print the matching chunk text here
            for span in gold_spans:
                matching_chunk = next((hit["text"] for hit in results if span in hit["text"]), None)
                if matching_chunk:
                    print(f"   Matching chunk: {matching_chunk.replace('\n', ' ')}")  



import pandas as pd
from typing import List, Dict
def analyze_retrieval_failures(
    rsys_map: Dict[str, RetrievalSystem],
    qa_df: pd.DataFrame,
    top_k: int = 5,
    relaxed: bool = True,
    retrieval_modes: List[str] = ["vector", "bm25", "hybrid", "expand"]
) -> pd.DataFrame:
    """
    Analizza i fallimenti (missed queries) per ogni combinazione di chunk type e retrieval mode.
    Restituisce un DataFrame con dettagli per debugging.
    """
    logs = []

    for chunk_type, rsys in rsys_map.items():
        print(f"\n[INFO] Analizzando chunk_type: {chunk_type}")
        for _, row in qa_df.iterrows():
            question = row["question"]
            gold_spans = row["answer_span"]

            for mode in retrieval_modes:
                if mode == "expand":
                    results = rsys.retrieve_and_expand(question, top_k=top_k, max_ref=2, mode="vector")
                else:
                    retriever = getattr(rsys, f"retrieve_{mode}")
                    results = retriever(question, top_k)

                # Check if match (relaxed or strict)
                if relaxed:
                    match = get_match_across_subspans(gold_spans, results)
                else:
                    match = any(span in r["text"] for r in results for span in gold_spans)

                if not match:

                    # Log record
                    logs.append({
                        "question": question,
                        "gold_spans": gold_spans,
                        "chunk_type": chunk_type,
                        "retrieval_mode": mode,
                        "match": match,
                        "top_k_ids": [r["id"] for r in results],
                        "top_k_scores": [r.get("score") for r in results],
                        "top_k_texts": [r["text"] for r in results],  # Optional truncation
                    })

    return pd.DataFrame(logs)


# 3) Funzione di evaluation per un singolo RetrievalSystem e modalità
def evaluate_method_on_qa(rsys, qa_df, mode: str, k: int = 5):
    total = len(qa_df)
    hits = 0
    scores_matched = []
    for _, row in qa_df.iterrows():
        q = row["question"]
        gold_spans = row["answer_span"]  # lista di sub-span
        # 1) recupero top-K
        if mode == "vector":
            results = rsys.retrieve_vector(q, k=k)
        elif mode == "bm25":
            results = rsys.retrieve_bm25(q, k=k)
        elif mode == "hybrid":
            results = rsys.retrieve_hybrid(q, k=k)
        elif mode == "expand":
            # usa retrieve_and_expand: top_k iniziale=k, max_ref=2 (o diverso se vuoi)
            results = rsys.retrieve_and_expand(q, top_k=k, max_ref=2, mode="vector")
        else:
            raise ValueError(f"Mode sconosciuto: {mode}")

        # 2) verifica relaxed match
        matched = get_match_across_subspans(gold_spans, results)
        if matched:
            hits += 1
            # raccogli punteggi dei chunk matching
            match_scores = []
            for chunk in results:
                for sub in gold_spans:
                    # per semplicità: controllo substring (potresti usare relaxed_span_match se preferisci)
                    if sub in chunk["text"] or sub.lower() in chunk["text"].lower():
                        if chunk.get("score") is not None:
                            match_scores.append(chunk["score"])
                        break
            if match_scores:
                scores_matched.append(np.mean(match_scores))
    recall = hits / total
    avg_score_hit = np.mean(scores_matched) if scores_matched else None
    return {"recall": recall, "avg_score_hit": avg_score_hit}


def evaluate_all_methods_on_qa(qa_df, chunk_types, k=5):
    rows = []
    for chunk_type, paths in chunk_types.items():
        print(f"\n>>> Evaluating chunk type: {chunk_type}")
        # Istanzia RetrievalSystem con i path corretti
        rsys = RetrievalSystem(
            faiss_index_path=paths["index_path"],
            meta_path=paths["meta_path"],
            vecs_path=paths["vecs_path"],
            graph_path=paths["graph_path"],
            embedding_model_name="all-MiniLM-L6-v2"
        )
        # Definiamo le modalità da testare: vector, bm25, hybrid; aggiungiamo 'expand' solo se reference_graph
        modes = ["vector", "bm25", "hybrid"]
        if chunk_type == "reference_graph":
            modes.append("expand")
        for mode in modes:
            print(f"- modalità {mode} ...")
            res = evaluate_method_on_qa(rsys, qa_df, mode=mode, k=k)
            rows.append({
                "chunk_type": chunk_type,
                "mode": mode,
                "recall": res["recall"],
                "avg_score_hit": res["avg_score_hit"]
            })

    # 5) Costruisci DataFrame riepilogo e pivot per recall
    results_df = pd.DataFrame(rows)
    pivot_recall = results_df.pivot(index="chunk_type", columns="mode", values="recall")
    pivot_score = results_df.pivot(index="chunk_type", columns="mode", values="avg_score_hit")

    print("\nRecall@K pivot table:")
    print(pivot_recall)
    print("\nAvg matched-score pivot table:")
    print(pivot_score)


def count_tokens(text):
    """Conta il numero di token usando nltk.word_tokenize."""
    return len(word_tokenize(text))


def evaluate_method_on_qa_extended(rsys, qa_df, mode: str, ks: list = [1, 3, 5, 10]):
    results_by_k = {}
    for k_val in ks:
        total = len(qa_df)
        hits = 0
        scores_matched = []
        avg_tokens_per_query = []
        retrieval_times = []

        for _, row in qa_df.iterrows():
            q = row["question"]
            gold_spans = row["answer_span"]  # lista di sub-span

            # ⏱️ Inizio tempo
            start = time.time()

            # 1) recupero top-K
            if mode == "vector":
                results = rsys.retrieve_vector(q, k=k_val)
            elif mode == "bm25":
                results = rsys.retrieve_bm25(q, k=k_val)
            elif mode == "hybrid":
                results = rsys.retrieve_hybrid(q, k=k_val)
            elif mode == "hybrid_rrf":
                results = rsys.retrieve_hybrid_rrf(q, k=k_val)
            elif mode == "expand":
                results = rsys.retrieve_and_expand(q, top_k=k_val, max_ref=2, mode="vector")
            else:
                raise ValueError(f"Mode sconosciuto: {mode}")

            # ⏱️ Fine tempo
            elapsed = time.time() - start
            retrieval_times.append(elapsed)

            # Token count
            current_query_tokens = sum(count_tokens(chunk["text"]) for chunk in results)
            if results:
                avg_tokens_per_query.append(current_query_tokens)

            # Matching
            print("Evaluation Match for question: ")
            print(q)
            matched = get_match_across_subspans(gold_spans, results)
            if matched:
                hits += 1
                match_scores = []
                for chunk in results:
                    for sub in gold_spans:
                        if sub in chunk["text"] or sub.lower() in chunk["text"].lower():
                            if chunk.get("score") is not None:
                                match_scores.append(chunk["score"])
                            break
                if match_scores:
                    scores_matched.append(np.mean(match_scores))

        recall = hits / total
        avg_score_hit = np.mean(scores_matched) if scores_matched else None
        mean_total_tokens = np.mean(avg_tokens_per_query) if avg_tokens_per_query else 0
        mean_retrieval_time = np.mean(retrieval_times) if retrieval_times else 0

        results_by_k[k_val] = {
            "recall": recall,
            "avg_score_hit": avg_score_hit,
            "mean_total_tokens": mean_total_tokens,
            "mean_retrieval_time": mean_retrieval_time
        }
    return results_by_k