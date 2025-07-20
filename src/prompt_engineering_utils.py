from retrieval_utils import *
import re
from pprint import pprint
from dotenv import load_dotenv
import pickle
from typing import Dict, List
import numpy as np
from gemini_utils import *

env_path: str = "../.env"
load_dotenv(env_path)


ALLOWED_SUBJECTS = {
    "AI_Act": "the EU AI Act",
    "A_Robust_Governance_for_the_AI_Act": "how to enforce Data Governcance in the EU AI Act",
    "GDPR": "the GDPR law"
}


def build_prompt(chunks, query, subject_list):
    # {"AI_Act", "A_Robust_Governance_for_the_AI_Act", "GDPR"}
    documents_prompt = ""
    for subject in subject_list:
        documents_prompt += f"{ALLOWED_SUBJECTS[subject]}, "
    # removing last comma and space
    documents_prompt = documents_prompt[:-2]

    intro = f""""
        You are a legal assistant helping users understand {documents_prompt}.
        Below is a question and some context extracted from official documents.
        Use only the provided context to answer, and cite relevant Articles or Annexes if possible.
        If the provided context does not contain enough information, respond with "I don't know".

        Context:
    """
    context_lines = []
    for chunk in chunks:
        text = chunk["text"].strip().replace("\n", " ")
        context_lines.append(f" {text}")
    
    question_block = f"\n\n---\n\nQuestion: {query}\n\nAnswer:"
    
    full_prompt = intro + "\n\n".join(context_lines) + question_block

    print("\n[INFO] Prompt completo:")
    print("-" * 80)
    print(full_prompt)
    print("-" * 80)

    return full_prompt


def ask_gemini_RAG(query, mode="vector", chunk_type="recursive", rsys=None, k=10):
    # getting the subject:
    subject_list = choose_subject_agent(query)
    
    # Re-Ranking
    final_top = k + (len(subject_list) - 1)*2
    final_top_chunks = get_final_rank(query, subject_list, chunk_type, mode=mode, final_top=final_top)

    full_prompt = build_prompt(final_top_chunks, query, subject_list)
    response_text = generate_gemini_content(full_prompt)

    return response_text, final_top_chunks


def split_text_by_subquestion(text):
    split_pattern = r'subquestion_\d+'
    subtexts = re.split(split_pattern, text)
    subtexts = [s.strip() for s in subtexts if s.strip()]
    return subtexts        


#### IRCoT-Style Reasoning Agent
def run_reasoning_agent(query: str, rsys: RetrievalSystem=None, model="gemini", K=10):
    
    n_api_calls = 0
    
    # 1. Usa Gemini per scomporre `query` in sub-domande
    decomposition_prompt = (
        "You are a legal assistant. Your task is to break down the following legal question into 2–4 smaller sub-questions. "
        "Each sub-question should be clear and help answer the original question step by step.\n\n"
        "Each sub-question should start with 'subquestion_N' where N is the number of the subquestion.\n\n"
        "Do not include any further element than 'subquestion_N' followed by the actual subquestion.\n\n"
        f"Main question: {query}\n\nSub-questions:"
    )

    sub_questions_text = generate_gemini_content(decomposition_prompt)
    n_api_calls += 1
    print(f"sub_questions_text \n\n {sub_questions_text}")
    sub_questions_list = split_text_by_subquestion(sub_questions_text)

    n_sub_questions=len(sub_questions_list)
    print(f"NUMBER OF SUB-QUESTIONS: {n_sub_questions}")

    print("Sub-questions generated:")
    pprint(sub_questions_list)

    ### STEP 2 — Retrieval e risposta per ogni sotto-domanda
    sub_answers = []
    all_chunks = []
    for sub_q in sub_questions_list:
        answer, chunks = ask_gemini_RAG(sub_q)
        print(f"SUB-QUESTION: {sub_q}")
        print(f"SUB-ANSWER: {answer}")
        sub_answers.append((sub_q, answer))
        all_chunks.extend(chunks)

    n_api_calls += n_sub_questions*2
    all_chunks = list(all_chunks)

    ### STEP 3 — Sintesi della risposta finale
    synthesis_prompt = f'''
        You are a legal assistant. Given the following question, and a set of sub-questions with their answers, 
        generate a clear, complete and coherent final answer, and cite relevant Articles or Annexes if possible.
        Use only the information provided by the sub-questions with their answers as knowledge base to elaborate the final answer.
        If the provided context does not contain enough information, respond with "I don't know". Do not hallucinate.\n\n
        Original Question: {query}\n\n
        Sub-questions and sub-answers:\n
        '''
    
    for i, (sub_q, ans) in enumerate(sub_answers, 1):
        synthesis_prompt += f"{i}. Sub-question: {sub_q}\n   Sub-answer: {ans.strip()}\n\n"


    print(f"FINAL SYNTHESIS PROMPT: {synthesis_prompt}")

    final_answer = generate_gemini_content(synthesis_prompt)
    n_api_calls+=1

    return final_answer.strip(), all_chunks, n_api_calls


def retrieve_citations(citations):
    ID_TO_META_INDEX_PATH = os.getenv("ID_TO_META_INDEX")

    with open(ID_TO_META_INDEX_PATH, "r", encoding="utf-8") as f:
        document_json_map = json.load(f)

    chunk_list = []           
    for id in citations:
        if id in document_json_map.keys():
            chunk_list.append(document_json_map[id])
        else:
            # take only the first keys that starts with: {id}.part_
            for key in document_json_map.keys():
                if ".part_" in key:
                    first = key.split(".part_")[0]
                    if first == id:
                        chunk_list.append(document_json_map[key])
                        continue

    return chunk_list


def get_citations(answer):
    prompt = f'''
        You are an extractor.  
        Below is an answer you generated to a legal question about the AI Act.  
        Your task: identify **all** AI Act Articles and Annexes that are **cited** in the answer or that are **most relevant**.  
        Produce as output a string containing a list of articles and annexes id separated by a comma ",", each with this structure:

        art_10, anx_XII, ...
        

        **Requirements:**
        1. Only include items actually mentioned or clearly implied in the answer.  
        2. Use the exact chunk IDs (e.g. `art_26`, `anx_III`).  

        ---

        **Here an example of answer you generated:**  
        > “Deployers of high-risk AI systems must ensure that…”  
        > “…as per Article 26, they shall perform a conformity assessment…”  
        > “…referring to Annex III definitions of high-risk categories…”  
        > “…and following the procedures described in Article 43.”  

        **Here an example of the list you have to output:** 

        art_26, anx_III, art_43

        **Here the answer where you have to extract the citations as explained above:**
        {answer}
            
    '''

    raw_citations = generate_gemini_content(prompt).strip()

    print("raw_citations: ")
    print(raw_citations)

    splitted_list = raw_citations.split(",")

    citations = []
    for citation in splitted_list:
        citations.append(citation.strip())

    citations = list(set(citations))

    retrieve_citations(citations)


def evaluate_answer(question, 
                    context_chunks, 
                    answer, 
                    correct_answer=False):
    
    additional_prompt = ""

    if correct_answer:
        with open("../data/benchmarks/qa_benchmark.json", "r", encoding="utf-8") as f:
            qa_benchmark = json.load(f)
        
        correct_answer_value = None
        for question_block in qa_benchmark:
            if question_block["question"] == question:
                correct_answer_value = question_block["answer"]

        if correct_answer_value:
            print("Adding correct answer benchmark")

            additional_prompt = f'''
                Here you can find a possible formulation of the correct answer to the 
                previous question that you can use as benchmark to evaluate the 
                answer generated by the legal AI System:
                
                "{correct_answer_value}"
            '''
        else:
            raise Exception(f"AnswerNotFoundException in qa_benchmark for question: {question}")
        
    
    prompt = f'''
        You are an expert evaluator of legal AI systems.
        A user asked the following question:
        "{question}"

        The legal AI system retrieved the following legal context chunks:

        {'-'*40}
        {"\n---\n".join([c['text'] for c in context_chunks])}

        Basing on the retrieved chunks the legal AI system generated the following answer:

        "{answer}"

        {additional_prompt}

        Please evaluate the answer according to these criteria:
            1. Groundedness: is the answer well-supported by the retrieved chunks?
            2. Relevance: does it directly address the user’s question?
            3. Structure: is it clearly written and legally precise?
            4. Hallucinations: does it invent articles or rules not found in the context?
        
        Provide a brief justification and rate each category from 1 (poor) to 5 (excellent).
    
    '''

    return generate_gemini_content(prompt)


def parse_evaluation_report(report: str) -> Dict[str, int]:

    scores = {}
    # Definiamo le metriche da cercare e come vogliamo mappare i loro nomi
    metrics = {
        "Groundedness": "groundedness",
        "Relevance": "relevance",
        "Structure": "structure",
        "Hallucinations": "hallucinations"
    }
    pattern = rf"(\d+)\s*/\s*\d+" # rf"\*\*\s*{pretty}\s*\*\*:\s*(\d+)\s*/\s*\d+"
    matches = re.findall(pattern, report)
    
    for pretty, score in zip(metrics.keys(), matches):
        # Cerca, ad esempio, "**Groundedness:** 3/5"
        
        if score:
            scores[pretty] = int(score)
        else:
            scores[pretty] = None

    return scores


def plan_refinement(eval_report: str) -> list[str]:
    prompt = f'''
        You are a legal‐QA coach.  The evaluation report below lists strengths and
        missing elements in an answer about the AI Act.  Identify _only_ the core
        _information gaps_ mentioned, and output them as a list of string separated by commas.
        Each string should be 2–4 words, e.g. risk identification, ongoing monitoring, ...

        Please provide ONLY the list of string separated by commas and nothing else.

    '''

    raw = generate_gemini_content(prompt)
    gaps_list = raw.split(",")
    return [x.strip() for x in gaps_list]


def get_sources_tool(query, answer, eval_report):
    prompt = f'''
        You are a legal‐QA coach and you are analysing the following couple of question 
        and answer:
        
        Question:
        {query}\n\n
    
        Answer:
        {answer}\n\n
        
        The evaluation report below lists strengths and missing elements in the answer 
        with respect to the given question.  

        Here the evaluation report:
        {eval_report}

        Your task is to output a comma separated list of articles and annexes that are needed to properly answer the question.

        Here an example of the output, in the case you need article 11 and annex III:
        
        art_11, anx_X
    )
    '''

    raw = generate_gemini_content(prompt)
    print("List of articles and annexes: ")
    print(raw)
    
    sources_list = raw.split(",")
    
    sources_list_clean = [str(x.strip()) for x in sources_list]

    print(f"Number of sources: {len(sources_list_clean)}")

    retrieved_citations = retrieve_citations(sources_list_clean)

    text_lean_list = []
    for x in retrieved_citations:
        new_dict = {}
        new_dict["text"] = x["text_lean"]
        text_lean_list.append(new_dict)

    return text_lean_list



def fetch_gap_chunks(
    gaps: List[str],
    article_dict: Dict[str, dict],
    chunk_vecs: Dict[str, np.ndarray],
    embedder,
    max_per_gap: int = 2
) -> List[dict]:
    
    extra_chunks = []
    
    # Build a list of article IDs & matrix for fallback search
    art_ids = list(article_dict.keys())
    art_texts = [article_dict[cid]["text_lean"] for cid in art_ids]
    art_embs  = embedder.encode(art_texts, convert_to_numpy=True).astype("float32")
    art_norms = np.linalg.norm(art_embs, axis=1)


    for gap in gaps:
        found = []
        kw = gap.lower()

        # 1) Keyword match first
        for cid, chunk in article_dict.items():
            if kw in chunk["text_lean"].lower():
                found.append(chunk)
                if len(found) >= max_per_gap:
                    print("Used Keyword match")
                    break
        
        # 2) Fallback to vector similarity if none found
        if not found:
            qv = embedder.encode([gap], convert_to_numpy=True).astype("float32")[0]
            qnorm  = np.linalg.norm(qv) + 1e-8
            # cosine sims against precomputed art_embs
            sims   = (art_embs @ qv) / (art_norms * qnorm + 1e-8)

            # pick top-max_per_gap
            top_k  = np.argsort(sims)[::-1][:max_per_gap]
            for idx in top_k:
                found.append(article_dict[art_ids[idx]])
                print("Used Fallback to vector similarity")

        extra_chunks.extend(found)

    return extra_chunks


def choose_subject_agent(question):
    prompt = f'''
        You are an LLM Agent expert in understanding the subjects of a question.
        
        The question that you have to analyse can belong to one, two or all of the following subjects:
            - AI_Act
            - A_Robust_Governance_for_the_AI_Act
            - GDPR

        Here the question:
        {question}

        Your task is to analyse the question and output a comma-separated list of strings containing
        the subset of subjects of the question.
        In particular:
            - include AI_Act into the list if the question is about, at least in part, the AI Act
            - include A_Robust_Governance_for_the_AI_Act into the list if the question is about, at least in part, the enforcement of Governance principles described by the AI Act
            - include GDPR into the list if the question is about, at least in part, the GDPR

        You output must be one of the possible combinations of subjects:
            "AI_Act"
            "GDPR"
            "A_Robust_Governance_for_the_AI_Act"
            "AI_Act, A_Robust_Governance_for_the_AI_Act"
            "AI_Act, GDPR"
            "GDPR, A_Robust_Governance_for_the_AI_Act"

        Please, output ONLY the comma-separated list of strings, DO NOT add other element in the output.

    '''

    raw_list = generate_gemini_content(prompt)
    print("List of question SUBJECTS: ")
    print(raw_list)
    
    ##### CLEANING PART
    # 1) Rimuovo eventuali backtick o virgolette
    cleaned = raw_list.strip().strip('`"\' \n')
    # 2) Split su virgola
    parts = [p.strip() for p in cleaned.split(',')]
    # 3) Elimino elementi vuoti
    subjects = [p for p in parts if p]
    
    fallback = False
    for s in subjects:
        if s not in ALLOWED_SUBJECTS.keys():
           print(f"NOT ALLOWED SUBJECT: {s!r} (from raw output: {raw_list!r})")
           print("Activatin FallBack")
           fallback = True
    
    if not fallback:
        return subjects
    else:
        subjects_fallback = []
        for s in subjects:
            if "ai act" in s.lower() or "AI_Act" in s:
                subjects_fallback.append("AI_Act")
            
            elif "GDPR" in s:
                subjects_fallback.append("AI_Act")

            elif "governance" in s.lower() or "A_Robust_Governance_for_the_AI_Act" in s:
                subjects_fallback.append("AI_Act")

            else:
                raise Exception("FAILED TO DETECT SUBJECTS")