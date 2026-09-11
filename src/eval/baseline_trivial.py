import json
import pandas as pd

def main():
    df = pd.read_csv("eval_data/golden_set.csv")
    
    # Majority class for intent
    majority_intent = df['intent'].mode()[0]
    intent_accuracy = (df['intent'] == majority_intent).mean()
    
    # Majority class for should_escalate
    majority_escalate = df['should_escalate'].mode()[0]
    escalate_accuracy = (df['should_escalate'] == majority_escalate).mean()
    
    metrics = {
        "model": "trivial_baseline",
        "description": "Predicts the majority class for all inputs.",
        "intent": {
            "majority_class": majority_intent,
            "accuracy": round(intent_accuracy, 4)
        },
        "should_escalate": {
            "majority_class": bool(majority_escalate),
            "accuracy": round(escalate_accuracy, 4)
        }
    }
    
    with open("eval_data/baseline_trivial.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Saved baseline metrics to eval_data/baseline_trivial.json")

if __name__ == "__main__":
    main()
