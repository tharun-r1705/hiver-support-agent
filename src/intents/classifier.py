import json
import pandas as pd
from src.llm_cache import cached_llm_call
import time

def classify_intent(text: str) -> str:
    prompt = f"""You are an expert customer support routing agent.
Analyze the following customer tweet and return ONLY a JSON object with one key "intent".
The intent must be one of: ["DELIVERY_DELAY", "MISSING_PACKAGE", "ACCOUNT_PAYMENT_ISSUE", "CANCELLATION_REFUND", "PRODUCT_INQUIRY", "POSITIVE_FEEDBACK", "OUT_OF_SCOPE"].

Customer Tweet:
{text}

Respond ONLY with valid JSON.
"""
    try:
        response = cached_llm_call(prompt)
        start = response.find("{")
        end = response.rfind("}") + 1
        json_str = response[start:end]
        data = json.loads(json_str)
        return data.get("intent", "OUT_OF_SCOPE")
    except Exception:
        return "OUT_OF_SCOPE"

def evaluate():
    print("Evaluating Intent Classifier against golden set...")
    df = pd.read_csv("eval_data/golden_set.csv")
    
    correct = 0
    total = len(df)
    
    for _, row in df.iterrows():
        pred = classify_intent(row["customer_text"])
        if pred == row["intent"]:
            correct += 1
            
        time.sleep(0.1)
            
    accuracy = correct / total
    
    metrics = {
        "model": "llm_intent_classifier",
        "description": "Zero-shot LLM classification of intent.",
        "intent": {
            "accuracy": round(accuracy, 4)
        }
    }
    
    with open("eval_data/eval_intent.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"LLM Intent Classifier Accuracy: {accuracy:.4f}")
    print("Saved evaluation metrics to eval_data/eval_intent.json")

if __name__ == "__main__":
    evaluate()
