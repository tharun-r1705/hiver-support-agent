import json
import pandas as pd
from src.llm_cache import cached_llm_call
import time

def decide_escalation(text: str) -> dict:
    prompt = f"""You are a customer support routing agent.
Analyze the following customer tweet and decide if it needs to be escalated to a human agent.
Usually, issues like missing packages, account/payment problems, and refunds require escalation because they need secure account access. Simple product inquiries or delivery delays often do not unless they are highly frustrated.

Customer Tweet:
{text}

Respond ONLY with valid JSON containing two keys:
- "should_escalate": boolean true/false
- "escalation_reason": string explaining why (or empty if false)
"""
    try:
        response = cached_llm_call(prompt)
        start = response.find("{")
        end = response.rfind("}") + 1
        json_str = response[start:end]
        data = json.loads(json_str)
        if isinstance(data.get("should_escalate"), bool):
            return data
        else:
            return {"should_escalate": False, "escalation_reason": "Invalid boolean format"}
    except Exception:
        return {"should_escalate": False, "escalation_reason": "Error parsing response"}

def evaluate():
    print("Evaluating Escalation Logic against golden set...")
    df = pd.read_csv("eval_data/golden_set.csv")
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    
    for _, row in df.iterrows():
        result = decide_escalation(row["customer_text"])
        pred_esc = result.get("should_escalate", False)
        actual_esc = bool(row["should_escalate"])
        
        if pred_esc and actual_esc:
            true_positives += 1
        elif pred_esc and not actual_esc:
            false_positives += 1
        elif not pred_esc and actual_esc:
            false_negatives += 1
        else:
            true_negatives += 1
            
        time.sleep(0.1)
            
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    metrics = {
        "model": "llm_escalation_decision",
        "description": "Zero-shot LLM classification for escalation.",
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4)
        }
    }
    
    with open("eval_data/eval_escalation.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("Saved evaluation metrics to eval_data/eval_escalation.json")

if __name__ == "__main__":
    evaluate()
