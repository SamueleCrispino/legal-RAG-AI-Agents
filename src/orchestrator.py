from prompt_engineering_utils import *
from retrieval_utils import *
import time

def rag_pipeline(question, 
                 chunk_type="recursive", 
                 mode="vector", 
                 k=10, 
                 fallback="hyde",
                 always_fallback=False):
    
    api_calls = 0
    pipe_start = time.time()

    # Direct Call ask_gemini_RAG    
    response_text, chunks = ask_gemini_RAG(question, mode=mode, chunk_type=chunk_type, k=k)
    api_calls+=2
    first_call_end = time.time()
    first_call_elapsed = first_call_end - pipe_start
    print("Direct Call ask_gemini_RAG: ")
    print(response_text)
    print()

    # EVALUATOR AGENT
    eval_report = evaluate_answer(question, chunks, response_text, correct_answer=True)
    api_calls+=1
    print("Evaluation report: ")
    print(eval_report)
    print()

    # REPORT PARSER AGENT
    scores_direct_call = parse_evaluation_report(eval_report)
    print("Parsed Scores: ")
    print(scores_direct_call)
    print()

    # FALLBACK
    if any(scores_direct_call.values()) < 4 or always_fallback:
        print("Activating FallBack: ", fallback)

        if fallback=='IRCoT':
            fallback_answer, fallback_chunks, n_api_calls_backoff = run_reasoning_agent(question)
        elif fallback=='hyde':
            fallback_answer, fallback_chunks = ask_gemini_RAG(question, mode=fallback, chunk_type=chunk_type, k=k)
            n_api_calls_backoff = 3
        else:
            raise ValueError(f"Unknown FallBack: {fallback}")
        
        fallback_end = time.time()
        fallback_elapsed = fallback_end - pipe_start

        api_calls += n_api_calls_backoff
        
        print("FallBack Answer: ")
        print(fallback_answer)
        print()

        # EVALUATOR AGENT
        eval_report = evaluate_answer(question, fallback_chunks, fallback_answer, correct_answer=True)
        print("Evaluation report: ")
        print(eval_report)
        print()

        # REPORT PARSER AGENT
        scores = parse_evaluation_report(eval_report)
        print("Parsed Scores: ")
        print(scores)
        print()

        return_payload = {
            "first_call_elapsed": first_call_elapsed,
            "fallback_elapsed": fallback_elapsed,
            "total_api_call": api_calls,
            "fallback": fallback
        }

        # Adding scores
        for score_name, score_value in scores.items():
            return_payload[f"fallback_{score_name}"] = score_value

        for score_name, score_value in scores_direct_call.items():
            return_payload[f"direct_{score_name}"] = score_value

        print("Returning Payload")
        print(return_payload)
        return return_payload
    
    return None


    ####
    # time
    # number of api call
    # score
