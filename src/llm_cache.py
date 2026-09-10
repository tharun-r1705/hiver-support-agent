import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

CACHE_DIR = Path("data/llm_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

def cached_llm_call(prompt: str) -> str:
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / f"{prompt_hash}.json"

    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("Using cached response...")
        return data["response"]

    response = llm.invoke(prompt)
    response_text = response.content

    cache_data = {"prompt": prompt, "response": response_text}
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)

    print("New response cached...")
    return response_text
