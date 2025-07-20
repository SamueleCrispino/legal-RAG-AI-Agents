import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # <-- Add this import

def build_and_persist_faiss_index_old(
    chunk_files_dir: str = "../data/chunks",
    model_name: str = "all-MiniLM-L6-v2",
    output_dir: str = "../data/indexes",
    batch_size: int = 64
):
    """
    Reads JSON chunk files, computes embeddings in batches, 
    builds a FAISS vector index, and saves both the index and metadata.

    Args:
        chunk_files_dir (str): Directory containing JSON chunk files.
        model_name (str): HuggingFace model name for SentenceTransformer.
        output_dir (str): Directory to store FAISS index and metadata.
        batch_size (int): Number of chunks to encode per batch.

    Produces:
        - FAISS index file named 'faiss_index_{model_name}.faiss'
        - Metadata pickle file named 'embeddings_meta_{model_name}.pkl'
    """

    # 1. Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load all chunks and their metadata
    # This metadata lets you map a retrieved vector (by its index ID in FAISS) back to:
    # Which chunking strategy produced it (source)
    # Which document it came from (all “ai_act” in this project)
    # The position of the chunk in the original document (offset)
    chunk_entries = []
    for fname in tqdm(os.listdir(chunk_files_dir), desc="Loading chunk files"):
        if not fname.endswith(".json"):
            continue
        method = fname.split("_")[2]  # expected pattern: ai_act_<method>_chunks_*.json
        path = os.path.join(chunk_files_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        for idx, text in enumerate(chunks):
            # Record each chunk with its source method and position index
            chunk_entries.append({
                "text": text,
                "source": method,
                "offset": idx
            })
    total_chunks = len(chunk_entries)
    print(f"[INFO] Loaded {total_chunks} chunks from '{chunk_files_dir}'.")

    # 3. Load the embedding model
    print(f"[INFO] Loading embedding model '{model_name}'...")
    encoder = SentenceTransformer(model_name)

    # 4. Encode chunks in batches
    print(f"[INFO] Encoding chunks in batches of {batch_size}...")
    embeddings = []
    for i in tqdm(range(0, total_chunks, batch_size), desc="Encoding batches"):
        batch_texts = [entry["text"] for entry in chunk_entries[i : i + batch_size]]
        batch_embs = encoder.encode(batch_texts, convert_to_numpy=True)
        embeddings.append(batch_embs)
    embeddings = np.vstack(embeddings).astype('float32')
    print(f"[INFO] Completed encoding. Final embedding matrix shape: {embeddings.shape}")

    # 5. Build FAISS index (Flat L2)
    # A single FAISS flat-L2 index that holds the dense embedding vectors for all chunks, 
    # regardless of which chunking strategy produced them.
    # FAISS stores a continuous array of vectors plus the index structure. 
    # It doesn’t by itself carry “metadata” about each vector’s origin
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    print(f"[INFO] Adding embeddings to FAISS index (dim={dim})...")
    index.add(embeddings)

    # 6. Persist index and metadata
    index_path = os.path.join(output_dir, f"faiss_index_{model_name}.faiss")
    meta_path  = os.path.join(output_dir, f"embeddings_meta_{model_name}.pkl")
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(chunk_entries, f)
    print(f"[INFO] Saved FAISS index to '{index_path}'")
    print(f"[INFO] Saved metadata to '{meta_path}'")
    
    # Return paths for convenience
    return index_path, meta_path

# Example usage:
# idx_path, meta_path = build_and_persist_faiss_index(
#     chunk_files_dir="chunks",
#     model_name="all-MiniLM-L6-v2",
#     output_dir="indexes",
#     batch_size=64
# )

def build_and_persist_index_per_chunk_file(
    chunk_json_dir: str = "../data/chunks",
    model_name: str = "all-MiniLM-L6-v2",
    output_dir: str = "../data/indexes",
    batch_size: int = 64
):
    """
    Per ogni file JSON in chunk_json_dir, carica i chunk, costruisce embeddings,
    crea un indice FAISS e salva:
      - faiss_index_{model_name}.faiss
      - chunks_meta_{model_name}.pkl
      - chunk_vecs_{model_name}.pkl
    in una sottocartella di output_dir basata sul nome del file (senza estensione).

    Questo permette di avere indici separati per ciascun metodo di chunking.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Carichiamo il modello di embedding una sola volta
    print(f"[INFO] Caricamento modello embedding '{model_name}'...")
    encoder = SentenceTransformer(model_name)

    # Iteriamo sui file JSON nella directory dei chunk
    for fname in sorted(os.listdir(chunk_json_dir)):
        if not fname.endswith(".json"):
            continue

        # Nome senza estensione, es. "ai_act_naive_chunks"
        base_name = os.path.splitext(fname)[0]
        sub_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(sub_output_dir, exist_ok=True)

        file_path = os.path.join(chunk_json_dir, fname)
        print(f"\n[INFO] Elaborazione file chunk: {fname}")
        # 1) Carica la lista di chunk dict
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        except Exception as e:
            print(f"[WARN] Impossibile leggere '{fname}': {e}")
            continue

        # 2) Filtra e prepara chunk_entries: devono essere dict con "id" e "text"
        chunk_entries = []
        for entry in chunks:
            if not isinstance(entry, dict) or "id" not in entry or "text" not in entry:
                print(f"[WARNING] Skip malformed entry in '{fname}': {entry}")
                continue
            e = dict(entry)  # shallow copy
            e["source_file"] = fname
            chunk_entries.append(e)

        total_chunks = len(chunk_entries)
        print(f"[INFO] Trovati {total_chunks} chunk validi in '{fname}'")
        if total_chunks == 0:
            print(f"[WARN] Nessun chunk valido in '{fname}', salto.")
            continue

        # 3) Encoding in batch
        embeddings_list = []
        chunk_vecs = {}
        print(f"[INFO] Encoding {total_chunks} chunk in batch size={batch_size}...")
        for i in tqdm(range(0, total_chunks, batch_size), desc=f"Encoding {base_name}"):
            batch = chunk_entries[i : i + batch_size]
            texts = [entry["text"] for entry in batch]
            embs = encoder.encode(texts, convert_to_numpy=True).astype("float32")
            embeddings_list.append(embs)
            for entry, vec in zip(batch, embs):
                cid = entry["id"]
                chunk_vecs[cid] = vec
        embeddings = np.vstack(embeddings_list)
        print(f"[INFO] Embedding matrix shape: {embeddings.shape}")

        # 4) Costruisci indice FAISS
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        print(f"[INFO] Aggiungo embeddings a FAISS index (dim={dim})...")
        index.add(embeddings)

        # 5) Persist in sub_output_dir
        idx_path   = os.path.join(sub_output_dir, f"faiss_index_{model_name}.faiss")
        meta_path  = os.path.join(sub_output_dir, f"chunks_meta_{model_name}.pkl")
        vecs_path  = os.path.join(sub_output_dir, f"chunk_vecs_{model_name}.pkl")

        faiss.write_index(index, idx_path)
        with open(meta_path, "wb") as f:
            pickle.dump(chunk_entries, f)
        with open(vecs_path, "wb") as f:
            pickle.dump(chunk_vecs, f)

        print(f"[INFO] Salvati in '{sub_output_dir}':")
        print(f"       - FAISS index: {os.path.basename(idx_path)}")
        print(f"       - Metadata:    {os.path.basename(meta_path)}")
        print(f"       - Chunk vecs:  {os.path.basename(vecs_path)}")

    print("\n[✅] Tutti gli indici per ciascun file di chunk sono stati creati.")

    # Restituiamo eventualmente una mappa di file creati, ma non strettamente necessario
    # Per esempio:
    # return { base_name: (idx_path, meta_path, vecs_path) for ciascun file }
