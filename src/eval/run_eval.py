import json
import pandas as pd
from src.llm_cache import cached_llm_call
import time
import os

def judge_reply(customer_text: str, ideal_reply: str, generated_reply: str) -> dict:
    prompt = f"""You are an expert customer support evaluator.
Evaluate the "Generated Reply" against the "Ideal Reply" for the given "Customer Tweet".
Rate it on a scale of 1 to 5, where 5 is perfect.

Customer Tweet:
{customer_text}

Ideal Reply:
{ideal_reply}

Generated Reply:
{generated_reply}

Respond ONLY with valid JSON containing:
- "score": integer 1-5
- "reasoning": brief explanation
"""
    try:
        response = cached_llm_call(prompt)
        start = response.find("{")
        end = response.rfind("}") + 1
        json_str = response[start:end]
        data = json.loads(json_str)
        return data
    except Exception:
        return {"score": 3, "reasoning": "Error parsing JSON, defaulting to 3"}

def run_end_to_end_eval():
    print("Running End-to-End Evaluation Harness (LLM-as-Judge)...")
    df = pd.read_csv("eval_data/generated_replies.csv")
    
    results = []
    
    for _, row in df.iterrows():
        judge_result = judge_reply(row["customer_text"], row["ideal_reply"], row["generated_reply"])
        row_dict = row.to_dict()
        row_dict["judge_score"] = judge_result.get("score", 3)
        row_dict["judge_reasoning"] = judge_result.get("reasoning", "")
        results.append(row_dict)
        time.sleep(0.1)
        
    res_df = pd.DataFrame(results)
    res_df.to_csv("eval_data/judge_results.csv", index=False)
    print(f"Average Judge Score: {pd.to_numeric(res_df['judge_score'], errors='coerce').mean():.2f} / 5")
    print("Saved judge results to eval_data/judge_results.csv")

if __name__ == "__main__":
    run_end_to_end_eval()
