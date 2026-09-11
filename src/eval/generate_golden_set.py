import os
import json
import pandas as pd
from src.llm_cache import cached_llm_call
import time

def generate_labels(text: str) -> dict:
    prompt = f"""You are a customer support labeler.
Analyze the following customer tweet and return a JSON object with four keys:
- "intent": one of ["DELIVERY_DELAY", "MISSING_PACKAGE", "ACCOUNT_PAYMENT_ISSUE", "CANCELLATION_REFUND", "PRODUCT_INQUIRY", "POSITIVE_FEEDBACK", "OUT_OF_SCOPE"]
- "ideal_reply": a short, helpful reply
- "should_escalate": boolean true/false
- "escalation_reason": string explaining why if true, else empty string

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
        return data
    except Exception as e:
        return {
            "intent": "OUT_OF_SCOPE",
            "ideal_reply": "I am sorry, please contact support.",
            "should_escalate": False,
            "escalation_reason": ""
        }

def main():
    df = pd.read_parquet("data/processed/amazonhelp_threads.parquet")
    sample_df = df.sample(n=150, random_state=42).copy()
    
    results = []
    for idx, row in sample_df.iterrows():
        text = row["customer_text"]
        label = generate_labels(text)
        label["thread_id"] = row["thread_id"]
        label["customer_text"] = text
        results.append(label)
        time.sleep(0.1)

    res_df = pd.DataFrame(results)
    os.makedirs("eval_data", exist_ok=True)
    res_df.to_csv("eval_data/golden_set.csv", index=False)
    print("Golden set generated at eval_data/golden_set.csv")

if __name__ == "__main__":
    main()
