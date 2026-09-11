import pandas as pd
from src.llm_cache import cached_llm_call
from src.retrieval.index import SimpleRetrievalIndex
import time
import os

class ReplyGenerator:
    def __init__(self, index_path: str):
        self.retrieval_index = SimpleRetrievalIndex(index_path)
        
    def generate_reply(self, customer_text: str) -> str:
        # Retrieve context
        retrieved_threads = self.retrieval_index.search(customer_text, top_k=2)
        
        context_str = ""
        for i, thread in enumerate(retrieved_threads, 1):
            context_str += f"\nExample {i}:\nCustomer: {thread['customer_text']}\nAmazonHelp: {thread['brand_reply']}\n"
            
        prompt = f"""You are an AmazonHelp customer support agent. 
Respond to the following customer tweet. Be empathetic, polite, and brief.
Use the provided examples of past interactions to match Amazon's brand voice.

{context_str}

Customer Tweet:
{customer_text}

Your Reply:
"""
        try:
            return cached_llm_call(prompt).strip()
        except Exception as e:
            return "I'm sorry to hear that. Please DM us your order details so we can assist you further."

def main():
    print("Generating grounded replies for the golden set...")
    generator = ReplyGenerator("data/processed/amazonhelp_threads.parquet")
    
    df = pd.read_csv("eval_data/golden_set.csv")
    
    results = []
    for idx, row in df.iterrows():
        customer_text = row["customer_text"]
        generated_reply = generator.generate_reply(customer_text)
        
        results.append({
            "thread_id": row["thread_id"],
            "customer_text": customer_text,
            "intent": row["intent"],
            "ideal_reply": row["ideal_reply"],
            "generated_reply": generated_reply
        })
        time.sleep(0.1)
        
    results_df = pd.DataFrame(results)
    os.makedirs("eval_data", exist_ok=True)
    results_df.to_csv("eval_data/generated_replies.csv", index=False)
    print("Saved generated replies to eval_data/generated_replies.csv")

if __name__ == "__main__":
    main()
