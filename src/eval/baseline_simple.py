import json
import pandas as pd

def predict_intent(text):
    text = str(text).lower()
    if any(w in text for w in ["stolen", "missing", "delivered but", "porch", "not received"]):
        return "MISSING_PACKAGE"
    if any(w in text for w in ["delay", "late", "where is", "hasn't updated", "promised", "when"]):
        return "DELIVERY_DELAY"
    if any(w in text for w in ["charge", "declined", "login", "password", "prime", "account", "payment"]):
        return "ACCOUNT_PAYMENT_ISSUE"
    if any(w in text for w in ["cancel", "refund", "return", "money back"]):
        return "CANCELLATION_REFUND"
    if any(w in text for w in ["stock", "availability", "kindle", "ps5", "video", "product", "item"]):
        return "PRODUCT_INQUIRY"
    if any(w in text for w in ["thanks", "love", "great", "good", "appreciate"]):
        return "POSITIVE_FEEDBACK"
    return "OUT_OF_SCOPE"

def predict_escalate(intent):
    return intent in ["MISSING_PACKAGE", "ACCOUNT_PAYMENT_ISSUE", "CANCELLATION_REFUND"]

def main():
    df = pd.read_csv("eval_data/golden_set.csv")
    
    intent_correct = 0
    escalate_correct = 0
    total = len(df)
    
    for _, row in df.iterrows():
        pred_intent = predict_intent(row["customer_text"])
        pred_esc = predict_escalate(pred_intent)
        
        if pred_intent == row["intent"]:
            intent_correct += 1
        if pred_esc == row["should_escalate"]:
            escalate_correct += 1
            
    metrics = {
        "model": "simple_heuristic_baseline",
        "description": "Rule-based keyword matching for intent and mapped escalation.",
        "intent": {
            "accuracy": round(intent_correct / total, 4)
        },
        "should_escalate": {
            "accuracy": round(escalate_correct / total, 4)
        }
    }
    
    with open("eval_data/baseline_simple.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Saved baseline metrics to eval_data/baseline_simple.json")

if __name__ == "__main__":
    main()
