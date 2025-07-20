import pickle
#!/usr/bin/env python
# coding: utf-8

# ## Naïve Fixed-Length Split
# 
# This method splits the document into chunks of fixed character length (e.g., 1000 characters), possibly with some overlap (e.g., 200 chars) to avoid breaking context mid-sentence.
# 
# ✅ Pros:
# - Simple to implement
# - Very fast
# - Useful baseline
# 
# ❌ Cons:
# - Can break sentences or paragraphs
# - May reduce semantic coherence
# - Poor performance on question answering over legal documents

# In[1]:


def chunk_naive(text: str, input_name: str, chunk_size=1000, overlap=200) -> list[dict]:
    """
    Splits `text` into fixed-length chunks (with overlap) and returns list of dicts:
      { "id": "<input_name>_naive_<i>", "text": "<chunk text>", "source": "naive", "offset": i }
    """
    chunks: list[dict] = []
    step = chunk_size - overlap
    idx = 0
    part = 0
    while idx < len(text):
        chunk_text = text[idx: idx + chunk_size]
        chunk_id = f"{input_name}_naive_{part}"
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "source": "naive",
            "offset": part
        })
        part += 1
        idx += step
    return chunks



# ## Recursive Split (LangChain-like)
# Breaks text using a hierarchy of delimiters (\n\n, \n, ., space). If the text is too long, it recursively attempts smaller splits until each piece fits the size limit:
# 
# "Split at paragraphs → if too long, split at sentences → if too long, split at words"
# 
# ✅ Pros:
# - Preserves more semantic boundaries
# - Less likely to break sentences or articles mid-way
# - More coherent context for LLM
# 
# ❌ Cons:
# - Slower than naïve
# - Still doesn’t understand semantic meaning — it uses structural heuristics

from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_recursive(text: str, input_name: str, chunk_size=1000, chunk_overlap=200) -> list[dict]:
    """
    Uses RecursiveCharacterTextSplitter to split text into strings, then wraps into dicts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    texts = splitter.split_text(text)
    chunks: list[dict] = []
    for i, chunk_text in enumerate(texts):
        chunk_id = f"{input_name}_recursive_{i}"
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "source": "recursive",
            "offset": i
        })
    return chunks



# ## Semantic Similarity Chunking
# Split text based on semantic meaning using sentence embeddings. You cluster or group sentences based on similarity or length budget to form semantically coherent chunks.
# - Group sentences that are close in meaning
# - Enforce a token or char budget per chunk
# 
# ✅ Pros:
# - Best for downstream question answering
# - Chunks align with conceptual boundaries
# - Reduces hallucination risk
# 
# ❌ Cons:
# - Slowest (requires embedding every sentence)
# - Requires a sentence transformer model
# - Slightly harder to implement/debug

from sentence_transformers import SentenceTransformer
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK sentence tokenizer (if not already downloaded)
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


def chunk_semantic_similarity(text, chunk_size=1000, similarity_threshold=0.75):
    """
    Chunk a text into semantically meaningful groups using sentence embeddings.
    
    Args:
        text (str): The input document (e.g., law text).
        chunk_size (int): Approximate maximum length of a chunk (in characters).
        similarity_threshold (float): Cosine similarity threshold to group sentences.

    Returns:
        List[str]: A list of semantically grouped text chunks.
    """

    # Ensure NLTK punkt is downloaded
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')


    # Load pre-trained sentence embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Split the text into individual sentences using NLTK
    sentences = sent_tokenize(text)

    # Compute vector embeddings for each sentence
    embeddings = model.encode(sentences)

    # Initialize output chunk list
    chunks = []

    # Start with the first sentence
    current_chunk = sentences[0]
    current_len = len(sentences[0])

    # Iterate over each sentence starting from the second one
    for i in range(1, len(sentences)):

        # Compute cosine similarity between current sentence and the previous one
        sim = cosine_similarity(
            [embeddings[i - 1]],  # embedding of previous sentence
            [embeddings[i]]       # embedding of current sentence
        )[0][0]  # cosine_similarity returns a matrix, extract scalar value

        # Get the length of the current sentence
        sentence_len = len(sentences[i])

        # Decision: Should we add this sentence to the current chunk?
        if sim >= similarity_threshold and (current_len + sentence_len) < chunk_size:
            # Semantically similar and still within length limit → group it
            current_chunk += " " + sentences[i]
            current_len += sentence_len
        else:
            # Otherwise, close current chunk and start a new one
            chunks.append(current_chunk.strip())
            current_chunk = sentences[i]
            current_len = sentence_len

    # Add the final chunk after the loop
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def chunk_semantic_dicts_wrapped(text: str, input_name: str, chunk_size=1000, similarity_threshold=0.75) -> list[dict]:
    """
    Splits text semantically into strings, then wraps into dicts.
    """
    str_chunks = chunk_semantic_similarity(text, chunk_size=chunk_size, similarity_threshold=similarity_threshold)
    chunks: list[dict] = []
    for i, chunk_text in enumerate(str_chunks):
        chunk_id = f"{input_name}_semantic_{i}"
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "source": "semantic",
            "offset": i
        })
    return chunks



import os
import json
from typing import List
from datetime import datetime
def chunk_text(text: str, method: str = "naive", input_name: str = "document", output_dir: str = "../data/chunks", **kwargs):
    """
    Now returns list of dicts for each chunk.
    """
    graph = None

    output_dir = output_dir + f"/{input_name}"

    os.makedirs(output_dir, exist_ok=True)

    if method == "naive":
        chunks = chunk_naive(text, input_name=input_name, **kwargs)
    elif method == "recursive":
        chunks = chunk_recursive(text, input_name=input_name, **kwargs)
    elif method == "semantic":
        chunks = chunk_semantic_dicts_wrapped(text, input_name=input_name, **kwargs)
    elif method == "reference_graph":
        chunks, graph = extract_ai_act_chunks_with_reference(
            html_path=text,
            max_tokens=kwargs.get("max_tokens", 2000),
            overlap_tokens=kwargs.get("overlap_tokens", 200)
        )
        # Note: extract_ai_act_chunks_with_reference already returns list[dict] with "id","text",...
    else:
        raise ValueError(f"Unknown chunking method: {method}")

    print(f"[✔] Method: {method} → Created {len(chunks)} chunks.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"{input_name}_{method}_chunks.json")

    # Save list of dicts
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[💾] Chunks saved to: {out_file}")

    if graph is None:
        return method, len(chunks)
    else:
        return method, len(chunks), graph


import re
import nltk
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from transformers import pipeline

def summarize_recital(text_to_summarize):
    # You might need to specify a model like "t5-small" or "t5-base"
    # or a fine-tuned summarization model.
    summarizer = pipeline("summarization", model="t5-small")

    # The prompt structure guides the summarization.
    # T5 often works well with "summarize: <text>"
    # For topic extraction, you can be more explicit.
    prompt = f"In 3-5 words, summarize this EU AI Act recital: {text_to_summarize}"

    summary_list = summarizer(prompt, max_length=5, min_length=3, do_sample=False)
    return summary_list[0]['summary_text']



import re
import nltk
from bs4 import BeautifulSoup
from typing import List, Dict

# ─────────────────────────────────────────────────────────────────────────────
# 1) Download NLTK punkt tokenizer (for subchunking)
# ─────────────────────────────────────────────────────────────────────────────
nltk.download('punkt', quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2) Helper: Build a consistent heading for each chunk
# ─────────────────────────────────────────────────────────────────────────────
def build_chunk_heading(chunk: dict) -> str:
    """
    Create a heading string that includes:
      - "<Type> <Number> – <Topic>"
      - Each non‐"No Chapter"/"No Section" ancestor on its own line
    Returns a heading terminated by two newlines.
    """
    lines = []

    # Main line: e.g. "Article 6 – Classification of high-risk AI systems"
    main_line = f"{chunk['type']} {chunk['number']} – {chunk['topic']}"
    lines.append(main_line)

    # Append each meaningful ancestor (chapter, section, etc.)
    for ancestor in chunk.get("ancestors", []):
        if ancestor:
            lines.append(ancestor)

    return "\n".join(lines) + "\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# 3) Subchunking helper
# ─────────────────────────────────────────────────────────────────────────────
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize

def subchunk_with_metadata(chunk: dict, max_tokens: int = 2000, overlap_tokens: int = 200) -> list[dict]:
    """
    Divide chunk["text"] in sotto‐chunk di al più max_tokens token (word_tokenize),
    con overlap overlap_tokens. Ogni sotto‐chunk:
      - mantiene id padre in 'id' con suffisso _p1, _p2, …
      - mantiene type, number, topic, ancestors, references
      - 'text' è heading + porzione reale di testo
    """
    text = chunk["text"]
    # Tokenizziamo tutto il testo
    tokens = word_tokenize(text)
    total = len(tokens)
    # Ricostruiamo heading (lo stesso usato originariamente)
    def build_chunk_heading(chunk):
        t = chunk.get("type", "")
        num = chunk.get("number", "")
        topic = chunk.get("topic", "")
        if t == "Recital":
            return f"Recital ({num})\n{topic}\n\n"
        elif t == "Article":
            return f"Article {num}\n{topic}\n\n"
        elif t in ["Chapter", "Section", "Annex"]:
            # es. "CHAPTER III – titolo"
            heading_type = t.upper() if t != "Article" else "Article"
            return f"{heading_type} {num} – {topic}\n\n"
        else:
            return f"{t} {num} – {topic}\n\n"
    heading = build_chunk_heading(chunk)
    heading_tokens = word_tokenize(heading)
    # Se non supera la soglia, ritorna un solo pezzo
    if total + len(heading_tokens) <= max_tokens:
        single = chunk.copy()
        single["sub_id"] = f"{chunk['id']}_p1"
        # Prepend heading se non già presente:
        if not chunk["text"].startswith(heading.strip()):
            single["text"] = heading + chunk["text"]
        return [single]
    # Altrimenti splittiamo
    pieces = []
    i = 0
    part = 0
    # free tokens per pezzo dopo heading
    free = max_tokens - len(heading_tokens)
    if free <= 0:
        raise ValueError("max_tokens troppo piccolo per contenere anche solo l'heading.")
    while i < total:
        part += 1
        # Seleziona porzione di token reali
        segment = tokens[i : i + free]
        # Ricostruisci testo del sotto‐chunk
        body = " ".join(segment)
        text_piece = heading + body
        new_meta = {
            **{k: chunk[k] for k in chunk if k not in ("text",)},  # copia metadata eccetto text
            "id":    f"{chunk['id']}.part_{part}",
            "sub_id": f"{chunk['id']}_p{part}",
            "text":  text_piece
        }
        # Se avevi references, mantieni lo stesso campo references:
        if "references" in chunk:
            new_meta["references"] = chunk["references"]
        new_meta["type"] = chunk.get("type")
        new_meta["number"] = chunk.get("number")
        new_meta["topic"] = chunk.get("topic")
        new_meta["ancestors"] = chunk.get("ancestors", [])
        pieces.append(new_meta)
        # Avanziamo con overlap
        i += free - overlap_tokens
    return pieces


def extract_references(text: str) -> Dict[str, List[str]]:
    """
    Returns a dict:
      {
        "articles": ["6", "43", ...],
        "annexes":  ["III", ...],
        # optionally "external_gdpr": ["35", ...], etc.
      }
    """
    ARTICLE_REF = re.compile(r"\bArticle\s+(\d+)")
    ANNEX_REF   = re.compile(r"\bAnnex\s+([IVXLCDM]+)")
    # (You could add patterns for “Recital \d+” or “GDPR Article \d+” if you want.)

    articles = re.findall(ARTICLE_REF, text)
    annexes  = re.findall(ANNEX_REF, text)
    return {
        "articles":  list(set(articles)),  # deduplicate
        "annexes":   list(set(annexes))
    }


import re
from typing import List, Dict
from bs4 import BeautifulSoup
from tqdm.auto import tqdm # Import tqdm for progress bars

# ─────────────────────────────────────────────────────────────────────────────
# 4) Main extractor function
# ─────────────────────────────────────────────────────────────────────────────
def extract_ai_act_chunks_with_reference(
    html_path: str,
    max_tokens: int = 2000,
    overlap_tokens: int = 200
) -> List[dict]:
    """
    Parses the AI Act HTML file at html_path and returns a list of chunk dictionaries.
    Keys in each chunk dict:
      - "id":         e.g. "rct_18", "cpt_III", "cpt_III.sct_4", "art_6", "annex_III"
      - "type":       one of ["Recital", "Chapter", "Section", "Article", "Annex"]
      - "number":     Recital → int, Chapter → roman/numeral, Section → int, Article → int, Annex → roman
      - "topic":      Recital: "recital X"
                      Chapter: "CHAPTER III – HIGH-RISK AI SYSTEMS"
                      Section: "SECTION 4 – Notifying authorities and notified bodies"
                      Article: from <p class="oj-sti-art">…</p>
                      Annex: from <p class="oj-doc-ti">…</p>
      - "ancestors": list of strings (e.g. ["Chapter III – …", "Section 4 – …"])
      - "text":       plain‐text of all <p> inside that div, prefixed by build_chunk_heading()
    Articles longer than max_tokens are split via subchunk_with_metadata().
    """
    # Dummy functions needed for the code to be runnable independently.
    # In your actual environment, these would be properly defined outside this function.
    def build_chunk_heading(chunk: dict) -> str:
        """Builds a heading for a chunk based on its type and topic."""
        type_map = {
            "Recital": "Recital",
            "Chapter": "CHAPTER",
            "Section": "SECTION",
            "Article": "Article",
            "Annex": "ANNEX"
        }
        heading_type = type_map.get(chunk["type"], "Unknown")
        heading_number = chunk.get("number", "")
        heading_topic = chunk.get("topic", "")

        if chunk["type"] == "Recital":
            return f"{heading_type} ({heading_number})\n{heading_topic}\n\n"
        elif chunk["type"] == "Article":
            return f"{heading_type} {heading_number}\n{heading_topic}\n\n"
        elif chunk["type"] in ["Chapter", "Section", "Annex"]:
            return f"{heading_type} {heading_number} – {heading_topic}\n\n"
        else:
            return f"{heading_type} {heading_number} - {heading_topic}\n\n"

    def extract_references(text: str) -> Dict[str, List[int]]:
        """
        Extracts references to articles and annexes from text.
        This is a simplified dummy implementation.
        """
        articles = re.findall(r"(?:Article|Articles)\s+(\d+)", text)
        annexes = re.findall(r"(?:Annex|Annexes)\s+([IVXLCDM]+)", text)
        return {
            "articles": [int(a) for a in articles],
            "annexes": annexes
        }

    def clean_text_inner(text: str | None) -> str:
        """Cleans up text, replacing None with an empty string."""
        
        return text if text is not None else ""

    # 4.1) Parse HTML
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    chunks: List[dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    # 4.2) Extract Recitals
    # ─────────────────────────────────────────────────────────────────────────
    recital_divs = soup.find_all("div", id=re.compile(r"^rct_\d+$"))
    print(f"\nExtracting {len(recital_divs)} Recitals...")
    for rec_div in tqdm(recital_divs, desc="Processing Recitals"):
        rec_id = rec_div["id"]          # "rct_18"
        rec_num = int(rec_id.split("_")[1])

        # Plain text: concatenate all <p> inside
        paras = rec_div.find_all("p")
        rec_text = "\n".join(p.get_text(strip=True) for p in paras)

        # Topic is simply "recital X"
        rec_topic = f"recital {rec_num}"

        # Build initial chunk and prepend heading
        chunk = {
            "id":        rec_id,
            "type":      "Recital",
            "number":    rec_num,
            "topic":     rec_topic,
            "ancestors": [],
            "text":      None,  # fill next
            "text_lean": None
        }
        chunk["text"] = build_chunk_heading(chunk) + rec_text
        refs = extract_references(chunk["text"])
        chunk["references"] = {
            "articles":  [f"art_{num}"    for num in refs["articles"]],
            "annexes":   [f"annex_{rome}" for rome in refs["annexes"]],
            # Expand to full IDs: “art_6”, “annex_III”
        }
        print("skipping recitals")
        continue
        chunks.append(chunk)

    # ─────────────────────────────────────────────────────────────────────────
    # 4.3) Build maps of Chapters and Sections (for ancestor lookup)
    # ─────────────────────────────────────────────────────────────────────────
    chapter_map: Dict[str, str] = {}
    section_map: Dict[str, str] = {}

    # 4.3.a) Chapters
    chapter_divs = soup.find_all("div", id=re.compile(r"^cpt_[IVXLCDM]+$"))
    print(f"\nBuilding map for {len(chapter_divs)} Chapters...")
    for chap_div in tqdm(chapter_divs, desc="Mapping Chapters"):
        chap_id = chap_div["id"]  # e.g. "cpt_III"

        # Label from <p class="oj-ti-section-1">
        label_p = chap_div.find("p", class_="oj-ti-section-1")
        label_text = label_p.get_text(strip=True) if label_p else chap_id

        # Title from <p class="oj-ti-section-2">
        title_p = chap_div.find("p", class_="oj-ti-section-2")
        title_text = title_p.get_text(strip=True) if title_p else ""

        full_chap_title = f"{label_text} – {title_text}" if title_text else label_text
        chapter_map[chap_id] = full_chap_title

    # 4.3.b) Sections
    section_divs = soup.find_all("div", id=re.compile(r"^cpt_[IVXLCDM]+\.sct_\d+$"))
    print(f"\nBuilding map for {len(section_divs)} Sections...")
    for sec_div in tqdm(section_divs, desc="Mapping Sections"):
        sec_id = sec_div["id"]  # e.g. "cpt_III.sct_4"

        # Label from <p class="oj-ti-section-1">
        label_p = sec_div.find("p", class_="oj-ti-section-1")
        label_text = label_p.get_text(strip=True) if label_p else sec_id

        # Title from <p class="oj-ti-section-2">
        title_p = sec_div.find("p", class_="oj-ti-section-2")
        title_text = title_p.get_text(strip=True) if title_p else ""

        full_sec_title = f"{label_text} – {title_text}" if title_text else label_text
        section_map[sec_id] = full_sec_title

    # ─────────────────────────────────────────────────────────────────────────
    # 4.4) Extract Articles (with heading + ancestors)
    # ─────────────────────────────────────────────────────────────────────────
    article_chunks: List[dict] = []
    article_divs = soup.find_all("div", id=re.compile(r"^art_\d+$"))
    print(f"\nExtracting {len(article_divs)} Articles...")
    for art_div in tqdm(article_divs, desc="Processing Articles"):
        art_id = art_div["id"]                          # e.g. "art_6"
        art_num = int(art_id.split("_")[1])             # 6

        # 4.4.a) Topic from <p class="oj-sti-art">
        title_p = art_div.find("p", class_="oj-sti-art")
        art_topic = title_p.get_text(strip=True) if title_p else f"Article {art_num}"

        # 4.4.b) Find ancestors by walking up the DOM
        chap_name = None
        sec_name = None
        parent = art_div.parent
        while parent:
            pid = parent.get("id", "")
            if pid in chapter_map and chap_name is None:
                chap_name = chapter_map[pid]
            if pid in section_map and sec_name is None:
                sec_name = section_map[pid]
            parent = parent.parent

        chap_name = clean_text_inner(chap_name)
        sec_name = clean_text_inner(sec_name)

        # 4.4.c) Extract plain‐text of all <p> inside this Article
        paras = art_div.find_all("p")
        art_text_body = "\n".join(p.get_text(strip=True) for p in paras)

        # Build full chunk dict, prepending heading
        chunk = {
            "id":        art_id,
            "type":      "Article",
            "number":    art_num,
            "topic":     art_topic,
            "ancestors": [chap_name, sec_name],
            "text":      None,  # filled next
            "text_lean": None
        }
        chunk["text"] = build_chunk_heading(chunk) + art_text_body
        chunk["text_lean"] = art_text_body

        ########## CREATING REFERENCES ##########
        refs = extract_references(chunk["text"])
        articles_ref_list = []

        # Looping in order to avoid self-reference !!
        for num in refs["articles"]:
            if f"art_{num}" != art_id:
                articles_ref_list.append(f"art_{num}")
        chunk["references"] = {
            "articles":  articles_ref_list,
            "annexes":   [f"annex_{rome}" for rome in refs["annexes"]],
            # Expand to full IDs: “art_6”, “annex_III”
        }

        # Add to article chunks
        article_chunks.append(chunk)

    # ─────────────────────────────────────────────────────────────────────────
    # 4.5) Extract Annexes (with heading)
    # ─────────────────────────────────────────────────────────────────────────
    annex_chunks: List[dict] = []
    annex_divs = soup.find_all("div", id=re.compile(r"^anx_[IVXLCDM]+$"))
    print(f"\nExtracting {len(annex_divs)} Annexes...")
    for annex_div in tqdm(annex_divs, desc="Processing Annexes"):
        ann_id = annex_div["id"]  # e.g. "annex_III"

        # Topic from <p class="oj-doc-ti">
        title_p = annex_div.find("p", class_="oj-doc-ti")
        ann_topic = title_p.get_text(strip=True) if title_p else ann_id

        paras = annex_div.find_all("p")
        ann_text_body = "\n".join(p.get_text(strip=True) for p in paras)

        chunk = {
            "id":        ann_id,
            "type":      "Annex",
            "number":    ann_id.split("_")[1],  # e.g. "III"
            "topic":     ann_topic,
            "ancestors": ["Annexes"],
            "text":      None,
            "text_lean": None
        }
        chunk["text"] = build_chunk_heading(chunk) + ann_text_body
        chunk["text_lean"] = ann_text_body

        refs = extract_references(chunk["text"])
        annexes_ref_list = []
        for rome in refs["annexes"]:
            if f"annex_{rome}" != ann_id:
                annexes_ref_list.append(f"annex_{rome}")
        # Avoid self-reference in articles
        chunk["references"] = {
            "articles":  [f"art_{num}" for num in refs["articles"]],
            "annexes":   annexes_ref_list,
            # Expand to full IDs: “art_6”, “annex_III”
        }

        annex_chunks.append(chunk)

    # ─────────────────────────────────────────────────────────────────────────
    # 4.6) Collect all chunks and subchunk Articles
    # ─────────────────────────────────────────────────────────────────────────
    all_chunks: List[dict] = []

    # 4.6.a) Add Recitals
    print(f"\n📥 Adding {len(chunks)} Recital chunks (no subchunking).")
    all_chunks.extend(chunks)

    # 4.6.b) Subchunk Articles if needed
    print(f"📥 Subchunking {len(article_chunks)} Articles if they exceed {max_tokens} tokens.")
    subchunked_articles_progress = tqdm(article_chunks, desc="Subchunking Articles")
    for art in subchunked_articles_progress:
        pieces = subchunk_with_metadata(
            chunk=art,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens
        )
        all_chunks.extend(pieces)
    print(f"Subchunking complete. Total chunks after subchunking: {len(all_chunks)}")


    # 4.6.c) Add Annexes
    print(f"📥 Adding {len(annex_chunks)} Annex chunks (no subchunking).")
    all_chunks.extend(annex_chunks)


    ##################################################################
    #################### CREATING THE REFERENCE GRAPH
    # Build a quick lookup: id → chunk dict
    chunk_by_id = {c["id"]: c for c in all_chunks}

    # Build adjacency graph
    # chunck_idX → [ref_id1, ref_id2, ...]
    graph = {}
    print("\nBuilding Reference Graph...")
    for c in tqdm(all_chunks, desc="Building Graph"):
        graph[c["id"]] = c["references"].get("articles", []) + c["references"].get("annexes", [])
    print("📊 Reference graph built.")



    with open("../data/indexes/document_reference_graph_chunks/graph_ai_act.pkl","wb") as f:
        pickle.dump(graph, f)


    print(f"\n✅ Extraction complete: {len(all_chunks)} total chunks.")
    return all_chunks, graph

