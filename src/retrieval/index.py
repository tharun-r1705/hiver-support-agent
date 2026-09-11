import pandas as pd
import re

class SimpleRetrievalIndex:
    def __init__(self, data_path: str):
        print(f"Loading data from {data_path}...")
        self.df = pd.read_parquet(data_path)
        print("Building index...")
        self.df['tokens'] = self.df['customer_text'].astype(str).apply(self._tokenize)
        print(f"Index built with {len(self.df)} records.")

    def _tokenize(self, text: str) -> set:
        return set(re.findall(r'\b\w+\b', text.lower()))

    def search(self, query: str, top_k: int = 3):
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        def jaccard(doc_tokens):
            intersection = len(query_tokens.intersection(doc_tokens))
            union = len(query_tokens.union(doc_tokens))
            return intersection / union if union > 0 else 0

        self.df['score'] = self.df['tokens'].apply(jaccard)
        top_results = self.df.nlargest(top_k, 'score')
        
        results = []
        for _, row in top_results.iterrows():
            results.append({
                'thread_id': row['thread_id'],
                'customer_text': row['customer_text'],
                'brand_reply': row['brand_reply'],
                'score': row['score']
            })
            
        return results

if __name__ == "__main__":
    index = SimpleRetrievalIndex("data/processed/amazonhelp_threads.parquet")
    sample_query = "Where is my package? It was supposed to be delivered today."
    print(f"\nSearching for: '{sample_query}'\n")
    results = index.search(sample_query, top_k=2)
    for i, res in enumerate(results, 1):
        print(f"Result {i} (Score: {res['score']:.4f}):")
        print(f"Customer: {res['customer_text']}")
        print(f"Brand: {res['brand_reply']}\n")
