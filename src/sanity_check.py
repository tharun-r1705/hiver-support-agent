from langchain_groq import ChatGroq
from llm_cache import cached_llm_call
from dotenv import load_dotenv

load_dotenv()
response = cached_llm_call("Health Check. Reply with 'Active'");
print(response)