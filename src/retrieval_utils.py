import os
import pickle
import numpy as np
import pandas as pd
import faiss                                    # FAISS per ricerca vettoriale
from sentence_transformers import SentenceTransformer  # Per codificare query
from rank_bm25 import BM25Okapi                # Per BM25 keyword search
from typing import List, Dict, Optional
from nltk.tokenize import word_tokenize
import re
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

from gemini_utils import *

def preprocess(text: str) -> List[str]:
    # 1) Normalizza NBSP
    text = text.replace('\xa0', ' ')
    # 2) Minuscolo
    text = text.lower()
    # 3) Tokenizza
    return word_tokenize(text)



class RetrievalSystem:
    """
    Interfaccia unificata di retrieval:
      - retrieve_vector: ricerca vettoriale via FAISS
      - retrieve_bm25: ricerca sparsa via BM25
      - retrieve_hybrid: combinazione pesata
      - expand_with_references: espansione basata sul grafo di riferimenti
      - retrieve_and_expand: retrieval + espansione
    """
    def __init__(
        self,
        faiss_index_path: str,
        meta_path: str,
        vecs_path: str,
        graph_path: Optional[str] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        # 1) Carica indice FAISS
        self.index = faiss.read_index(faiss_index_path)

        # 2) Carica metadata: lista di dict per ogni chunk
        #    Ogni dict deve contenere almeno: "id", "text", e può avere campi extra ("type","topic","ancestors",...)
        with open(meta_path, "rb") as f:
            self.meta: List[dict] = pickle.load(f)

        # 3) Costruisci mappa id -> metadata dict, per accesso diretto
        self.id_to_meta: Dict[str, dict] = {}
        for entry in self.meta:
            cid = entry.get("id")
            if cid is None:
                raise ValueError(f"Entry di metadata senza 'id': {entry}")
            self.id_to_meta[cid] = entry

        # 4) Prepara BM25: servono i testi tokenizzati
        #    Estrae il campo "text" da ogni entry in self.meta
        
        self.documents: List[str] = [entry["text"] for entry in self.meta]
        # Tokenizzazione semplice: split su whitespace. Se serve, si può migliorare con nltk/spaCy.
        tokenized_corpus = [preprocess(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 5) Carica modello di embedding per query encoding
        self.embedder = SentenceTransformer(embedding_model_name)

        # 6) Carica dizionario chunk_vecs: id_chunk -> vettore embedding numpy (float32)
        with open(vecs_path, "rb") as f:
            self.chunk_vecs: Dict[str, np.ndarray] = pickle.load(f)

        # 7) Carica grafo di riferimenti se fornito, altrimenti vuoto
        if graph_path:
            with open(graph_path, "rb") as f:
                self.graph: Dict[str, List[str]] = pickle.load(f)
        else:
            # se non fornito, lascia vuoto o chiedi di settarlo dall'esterno
            self.graph = {}

    def retrieve_vector(self, query: str, k: int = 5) -> List[dict]:
        """
        Ricerca vettoriale: codifica la query, interroga FAISS per k vicini.
        Ritorna lista di dict con almeno:
          - "id": chunk id
          - "text": chunk text
          - "score": distanza L2 (float) o similarità inversa
          - eventuali altri metadata (es. "type","topic","ancestors","source","offset")
        """
        # 1) Encode query in embedding space
        q_emb = self.embedder.encode([query]).astype("float32")  # shape (1, dim)

        # 2) FAISS search: restituisce (distances, indices)
        distances, indices = self.index.search(q_emb, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # idx è l'indice nel vettore di embedding = indice in self.meta
            entry = self.meta[idx]
            # Creiamo il dict di risultato includendo id e text e eventuali metadati
            res = {
                "id":    entry["id"],
                "text":  entry["text"],
                "score": float(dist)  # distanza L2: più piccola = più simile
            }
            # Se esistono altri campi utili, includili:
            # e.g.: type, topic, ancestors, source_file, offset...
            for key in ("type", "topic", "ancestors", "source", "source_file", "offset"):
                if key in entry:
                    res[key] = entry[key]
            results.append(res)
        return results

    def retrieve_bm25(self, query: str, k: int = 5) -> List[dict]:
        """
        Ricerca BM25 su tokenized_corpus.
        Ritorna lista di dict con stessi campi di retrieve_vector (id, text, score, ...).
        """
        tokenized_query = preprocess(query)
        scores = self.bm25.get_scores(tokenized_query)
        # ordiniamo indici per punteggio desc
        topk_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in topk_indices:
            entry = self.meta[idx]
            res = {
                "id":    entry["id"],
                "text":  entry["text"],
                "score": float(scores[idx])  # punteggio BM25
            }
            for key in ("type", "topic", "ancestors", "source", "source_file", "offset"):
                if key in entry:
                    res[key] = entry[key]
            results.append(res)
        return results

    def retrieve_hybrid(self, query: str, k: int = 5, alpha: float = 0.5) -> List[dict]:
        """
        Recupero ibrido: combina punteggi normalizzati da vettoriale e BM25.
        alpha=1.0 -> solo vettoriale; alpha=0.0 -> solo BM25.
        """
        # 1) Vector retrieval su tutti i chunk: otteniamo distanze e indici
        q_emb = self.embedder.encode([query]).astype("float32")
        distances, indices = self.index.search(q_emb, len(self.meta))
        # Convertiamo L2 distance in score (più grande = più simile): ad esempio -distance
        vec_scores = -distances[0]  # array shape (N,)
        # Normalizziamo in [0,1]
        vec_min, vec_ptp = vec_scores.min(), vec_scores.ptp()
        vec_scores = (vec_scores - vec_min) / (vec_ptp + 1e-8)

        # 2) BM25 su tutti i chunk
        tokenized_query = preprocess(query)
        bm25_raw = self.bm25.get_scores(tokenized_query)
        bm25_min, bm25_ptp = bm25_raw.min(), bm25_raw.ptp()
        bm25_scores = (bm25_raw - bm25_min) / (bm25_ptp + 1e-8)

        # 3) Combina
        hybrid_scores = alpha * vec_scores + (1 - alpha) * bm25_scores

        # 4) Prendi top-k
        topk_indices = np.argsort(hybrid_scores)[::-1][:k]

        results = []
        for idx in topk_indices:
            entry = self.meta[idx]
            res = {
                "id":    entry["id"],
                "text":  entry["text"],
                "score": float(hybrid_scores[idx])
            }
            for key in ("type", "topic", "ancestors", "source", "source_file", "offset"):
                if key in entry:
                    res[key] = entry[key]
            results.append(res)
        return results

    def retrieve_hybrid_rrf(self, query: str, k: int = 5, k_rrf: int = 60) -> List[dict]:
        """
            Hybrid retrieval using Reciprocal Rank Fusion (RRF).
            k_rrf = costante di smoothing (“offset”)
        """

        # 1) Get rankings from vector and BM25
        q_emb = self.embedder.encode([query]).astype("float32")
        distances, indices = self.index.search(q_emb, len(self.meta))
        vec_rank = indices[0]

        tokenized_query = preprocess(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_rank = np.argsort(bm25_scores)[::-1]  # higher is better

        # 2) Build dict: doc_id -> RRF score
        rrf_scores = {}

        def add_rrf_scores(rank_list):
            for rank, idx in enumerate(rank_list):
                doc_id = idx
                # con k_rrf > 0 tutte le frazioni diventano un po’ più piccole 
                # e “appiattiscono” la curva dei punteggi
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k_rrf + rank)

        add_rrf_scores(vec_rank)
        add_rrf_scores(bm25_rank)

        # 3) Sort by RRF score
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for idx, score in sorted_items:
            entry = self.meta[idx]
            res = {
                "id": entry["id"],
                "text": entry["text"],
                "score": score
            }
            for key in ("type", "topic", "ancestors", "source", "source_file", "offset"):
                if key in entry:
                    res[key] = entry[key]
            results.append(res)
        return results
    
    def expand_with_references(self, retrieved_ids: List[str], query: str, max_ref: int = 2, freq_weight: float = 1.0) -> List[str]:
        """
        Data una lista di retrieved_ids (top-N), raccoglie fino a max_ref riferimenti
        tramite il grafo self.graph, li ordina combinando:
        - similarità coseno con la query
        - frequenza di riferimento (quante volte compare nei riferimenti dei retrieved_ids)
        e ritorna i migliori max_ref IDs.
        freq_weight: peso della componente frequenza nel punteggio finale.
        """
        print(f"[DEBUG] Espansione con riferimenti per {len(retrieved_ids)} IDs: {retrieved_ids}")
        # 1) Aggrega referenze e conta frequenze
        candidate_refs = {}
        for rid in retrieved_ids:
            
            #set_refs = list(set(self.graph.get(rid, []))) # evita duplicati
            for ref_id in self.graph.get(rid, []):
                # Uniforma prefisso Annex se serve
                if "annex" in ref_id:
                    roman_number = ref_id.split("_")[1]
                    ref_id = "anx_" + roman_number

                if ref_id in self.id_to_meta:
                    candidate_refs[ref_id] = candidate_refs.get(ref_id, 0) + 1
                else:
                    # prova a trovare subchunks
                    subs = []
                    subs += [k for k in self.id_to_meta if k.startswith(ref_id + ".part_")]
                    subs += [k for k in self.id_to_meta if k.startswith(ref_id + "_p")]
                    if subs:
                        print(f"[DEBUG] Trovati subchunk per {ref_id}: {subs}")
                        for sub_id in subs:
                            candidate_refs[sub_id] = candidate_refs.get(sub_id, 0) + 1
                    else:
                        print(f"[WARNING] Riferimento {ref_id} non trovato in id_to_meta, né subchunk, salto.")
        print(f"[DEBUG] candidate_refs (with counts): {candidate_refs}")
        # 2) Rimuovi quelli già recuperati
        for rid in retrieved_ids:
            candidate_refs.pop(rid, None)
        if not candidate_refs:
            print(f"[DEBUG] Nessun riferimento candidato trovato dopo rimozione di IDs iniziali.")
            return []

        # 3) Trova max frequency per normalizzare
        max_freq = max(candidate_refs.values())
        for k, v in candidate_refs.items():
            print(f"[DEBUG] Riferimento {k} ha frequenza {v} (normalizzato: {v / max_freq:.4f})")

        # 4) Codifica la query una volta
        q_emb = self.embedder.encode([query]).astype("float32")[0]
        q_norm = np.linalg.norm(q_emb) + 1e-8

        # 5) Calcola punteggio combinato per ciascun candidate
        scored = []
        for ref_id, freq in candidate_refs.items():
            ref_vec = self.chunk_vecs.get(ref_id)
            if ref_vec is None:
                continue
            # cosine similarity
            sim = float(np.dot(q_emb, ref_vec) / (q_norm * (np.linalg.norm(ref_vec) + 1e-8)))
            # normalizza frequenza in [0,1]
            freq_norm = freq / max_freq
            # punteggio finale: somma pesata o moltiplicazione
            # opzione 1: weighted sum
            final_score = sim + freq_weight * freq_norm
            # se preferisci somma pesata con somma di coefficienti:
            # alpha = 1.0 / (1.0 + freq_weight)
            # final_score = alpha * sim + (1 - alpha) * freq_norm
            # opzione 2: boost multiplicativo
            # final_score = sim * (1 + freq_weight * freq_norm)
            scored.append((final_score, ref_id, sim, freq_norm))

        if not scored:
            return []
        # 6) Ordina per punteggio finale decrescente
        scored.sort(key=lambda x: x[0], reverse=True)
        top_refs = [ref_id for _, ref_id, _, _ in scored[:max_ref]]

        print(f"[DEBUG] Espansione candidati ordinati (score, sim, freq_norm):")
        for score, ref_id, sim, freq_norm in scored[:max_ref]:
            print(f"  {ref_id}: final_score={score:.4f}, sim={sim:.4f}, freq_norm={freq_norm:.4f}")

        return top_refs

    def retrieve_and_expand(self, query: str, top_k: int = 3, max_ref: int = 2, mode: str = "vector") -> List[dict]:
        """
        Esempio di flusso:
          1) Recupero iniziale via 'mode' in ["vector","bm25","hybrid"]
          2) Espansione tramite riferimenti
          3) Restituisce lista di chunk dict completi (inclusi metadata)
        """
        # 1) Retrieval iniziale
        if mode == "vector":
            hits = self.retrieve_vector(query, k=top_k)
        elif mode == "bm25":
            hits = self.retrieve_bm25(query, k=top_k)
        elif mode == "hybrid":
            hits = self.retrieve_hybrid(query, k=top_k)
        else:
            raise ValueError(f"Mode sconosciuto: {mode}")

        print(f"retrieved {len(hits)} hits iniziali con mode={mode}")
        retrieved_ids = [h["id"] for h in hits]
        print(f"[DEBUG] Recuperati {len(retrieved_ids)} IDs iniziali: {retrieved_ids}")

        # 2) Espandi con riferimenti
        ref_ids = self.expand_with_references(retrieved_ids, query=query, max_ref=max_ref)

        # 3) Costruisci lista completa di chunk dict: concateno hits + riferimenti
        final = hits.copy()
        for rid in ref_ids:
            entry = self.id_to_meta.get(rid)
            if entry:
                # aggiungo lo stesso formato dict di retrieve_vector
                res = {
                    "id":   entry["id"],
                    "text": entry["text"],
                    "score": None  # o lascia None: è un'espansione, non un punteggio diretto
                }
                for key in ("type", "topic", "ancestors", "source", "source_file", "offset"):
                    if key in entry:
                        res[key] = entry[key]
                final.append(res)
        return final

    def retrieve_hyde(self, query: str, k: int = 5, max_tokens: int = 200) -> List[dict]:
        """
        HyDE: Generate hypothetical answer → embed → retrieve similar chunks
        """
        # 1. Genera risposta ipotetica con LLM
        prompt = f"Answer this legal question concisely and precisely:\n\n{query}\n\nAnswer:"
        hypothetical_answer = generate_gemini_content(prompt)  # Usa Gemini o un altro LLM
        
        print("Hyde Hypothetical Answer: ")
        print(hypothetical_answer)
        
        # Optional: Truncate if too long
        if len(hypothetical_answer) > max_tokens:
            hypothetical_answer = hypothetical_answer[:max_tokens]

        
        return self.retrieve_vector(hypothetical_answer, k=k)



def get_paths(document_folder, chunk_type):
    base_indexes = f"../data/indexes/{document_folder}"
    chunk_types = {
        "naive": {
            "index_path": os.path.join(base_indexes, f"{document_folder}_naive_chunks", "faiss_index_all-MiniLM-L6-v2.faiss"),
            "meta_path":  os.path.join(base_indexes, f"{document_folder}_naive_chunks", "chunks_meta_all-MiniLM-L6-v2.pkl"),
            "vecs_path":  os.path.join(base_indexes, f"{document_folder}_naive_chunks", "chunk_vecs_all-MiniLM-L6-v2.pkl"),
            "graph_path": None
        },
        "recursive": {
            "index_path": os.path.join(base_indexes, f"{document_folder}_recursive_chunks", "faiss_index_all-MiniLM-L6-v2.faiss"),
            "meta_path":  os.path.join(base_indexes, f"{document_folder}_recursive_chunks", "chunks_meta_all-MiniLM-L6-v2.pkl"),
            "vecs_path":  os.path.join(base_indexes, f"{document_folder}_recursive_chunks", "chunk_vecs_all-MiniLM-L6-v2.pkl"),
            "graph_path": None
        },
        "semantic": {
            "index_path": os.path.join(base_indexes, f"{document_folder}_semantic_chunks", "faiss_index_all-MiniLM-L6-v2.faiss"),
            "meta_path":  os.path.join(base_indexes, f"{document_folder}_semantic_chunks", "chunks_meta_all-MiniLM-L6-v2.pkl"),
            "vecs_path":  os.path.join(base_indexes, f"{document_folder}_semantic_chunks", "chunk_vecs_all-MiniLM-L6-v2.pkl"),
            "graph_path": None
        },
        "reference_graph": {
            "index_path": os.path.join(base_indexes, "document_reference_graph_chunks", "faiss_index_all-MiniLM-L6-v2.faiss"),
            "meta_path":  os.path.join(base_indexes, "document_reference_graph_chunks", "chunks_meta_all-MiniLM-L6-v2.pkl"),
            "vecs_path":  os.path.join(base_indexes, "document_reference_graph_chunks", "chunk_vecs_all-MiniLM-L6-v2.pkl"),
            "graph_path": os.path.join(base_indexes, "document_reference_graph_chunks", "graph_{document_folder}.pkl")
        }
    }

    return chunk_types[chunk_type]


def federated_retrieve(question, subjects, chunk_type="recursive", mode="vector", k=10):
    """
    Per ogni subject:
      - istanzia RetrievalSystem
      - recupera top-k (vector, bm25, ecc.)
    Ritorna un dizionario id->(chunk, list_of_ranks)
    """
    all_hits = {}  # id -> {"chunk": chunk_dict, "ranks": [r1, r2, ...], "scores": [s1,s2,...]}
    for subject in subjects:
        paths = get_paths(subject, chunk_type)
        rsys = RetrievalSystem(
            faiss_index_path=paths["index_path"],
            meta_path=paths["meta_path"],
            vecs_path=paths["vecs_path"],
            graph_path=paths["graph_path"],
            embedding_model_name="all-MiniLM-L6-v2"
        )

        print(f"Using mode: {mode}")

        if mode == "vector":
            hits = rsys.retrieve_vector(question, k=k)  # o mode dinamico
        elif mode == "bm25":
            hits = rsys.retrieve_bm25(question, k=k)
        elif mode == "hybrid":
            hits = rsys.retrieve_hybrid(question, k=k)
        elif mode == "hybrid_rrf":
            hits = rsys.retrieve_hybrid_rrf(question, k=k)
        elif mode == "expand":
            hits = rsys.retrieve_and_expand(question, k=k)
        elif mode == "hyde":
            hits = rsys.retrieve_hyde(question, k=k)
        else:
            raise ValueError(f"Mode sconosciuto: {mode}")


        for rank, chunk in enumerate(hits, start=1):
            cid = chunk["id"]
            if cid not in all_hits:
                all_hits[cid] = {
                    "chunk": chunk,
                    "ranks": [],
                    "scores": []
                }
            all_hits[cid]["ranks"].append(rank)
            all_hits[cid]["scores"].append(chunk["score"])
    return all_hits


def rrf_rerank(all_hits, phi=60):
    reranked = []
    for cid, info in all_hits.items():
        # calcola RRF score
        rrf_score = sum(1.0 / (phi + r) for r in info["ranks"])
        # opzionalmente puoi anche mediare gli "original scores"
        avg_score = sum(info["scores"]) / len(info["scores"])
        # combini i due:
        final_score = rrf_score + avg_score  # o pesali diversamente
        reranked.append((final_score, info["chunk"]))
    # ordina descending
    reranked.sort(key=lambda x: x[0], reverse=True)
    # ritorna solo i chunk dict
    return [chunk for _, chunk in reranked]


def cross_rerank(question, candidates, top_m=50):
    # prendi i primi top_m dal RRF
    small_pool = candidates[:top_m]
    pairs = [[question, c["text"]] for c in small_pool]
    ce_scores = cross_encoder.predict(pairs)  # array di float
    scored = list(zip(ce_scores, small_pool))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


def get_final_rank(question, subjects, chunks_type, mode="vector", k=10, phi=60, final_top=10):
    hits_map = federated_retrieve(question, subjects, chunks_type, mode=mode, k=k)
    # 1) RRF fusion
    fused = rrf_rerank(hits_map, phi=phi)
    # 2) eventualmente cross‑rerank
    reranked = cross_rerank(question, fused, top_m= min(50, len(fused)))
    # 3) prendi i primi final_top
    
    final_rank = reranked[:final_top]
    final_rank_df = pd.DataFrame(final_rank)  
    final_rank_df["subject"] = final_rank_df.id.apply(lambda x: x.split(f"_{chunks_type}_")[0])
    
    print("FINAL RANK COMPOSITION: ")
    print(final_rank_df.groupby("subject").count().head())

    return reranked[:final_top]

