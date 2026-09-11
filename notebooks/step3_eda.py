import pandas as pd
import numpy as np

def run_eda():
    df = pd.read_parquet('../data/processed/amazonhelp_threads.parquet')
    
    # 1. Quantitative EDA
    df['customer_len'] = df['customer_text'].apply(lambda x: len(str(x)))
    df['reply_len'] = df['brand_reply'].apply(lambda x: len(str(x)))
    
    print("=== Quantitative EDA ===")
    print(f"Total threads: {len(df)}")
    print(f"Avg customer message length: {df['customer_len'].mean():.2f} chars")
    print(f"Avg brand reply length: {df['reply_len'].mean():.2f} chars")
    
    # Let's save 100 random threads for reading
    sample_df = df.sample(100, random_state=42)
    
    with open('sample_100_threads.txt', 'w', encoding='utf-8') as f:
        for idx, row in sample_df.iterrows():
            f.write(f"--- Thread {idx} ---\n")
            f.write(f"CUSTOMER: {row['customer_text']}\n")
            f.write(f"BRAND: {row['brand_reply']}\n\n")
            
    print("\nSaved 100 samples to notebooks/sample_100_threads.txt")

if __name__ == '__main__':
    run_eda()
