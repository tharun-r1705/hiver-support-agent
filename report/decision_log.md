# Decision Log

Throughout the development of the Hiver Support Agent, numerous architectural and methodological decisions were made. Below are 15 key judgment calls that guided the project:

### 1. Brand Selection
Selected `AmazonHelp` from the Kaggle dataset because it offered the highest volume of diverse, recognizable, and highly structured support interactions (e.g., missing items, delivery delays).

### 2. Subsampling Strategy
Subsampled the dataset to exactly 30,000 interactions. This size was chosen to provide enough diversity for the retrieval index while ensuring fast in-memory Pandas operations without requiring distributed computing frameworks like PySpark.

### 3. LLM Model Selection
Used `gpt-oss-120b` via the Groq API because it offered near-instantaneous inference speeds for zero-shot prompts, avoiding the latency bottlenecks typical of large parameter models.

### 4. Caching Mechanism
Implemented a file-based JSON cache (`src/llm_cache.py`) using SHA256 hashes of the prompts. This was critical to prevent redundant API calls during the iterative development of evaluation scripts, saving both time and API quotas.

### 5. Taxonomy Granularity
Restricted the intent taxonomy to 7 specific categories (e.g., `MISSING_PACKAGE`, `DELIVERY_DELAY`, `OUT_OF_SCOPE`) instead of 20+. This simplified the zero-shot classification task for the LLM and reduced label ambiguity.

### 6. Synthetic Golden Set Generation
Chose to leverage the LLM to synthetically generate the "golden set" labels (intent, ideal reply, escalation) for 150 rows. This accelerated development, though it introduced recognized synthetic bias into the evaluation metrics.

### 7. Trivial Baseline Definition
Defined the "trivial baseline" as the absolute majority class for both intent and escalation. This provided a necessary accuracy floor to ensure our classifiers were actually learning rather than just predicting the most common issue.

### 8. Simple Heuristic Baseline
Opted to write a pure Python rule-based keyword matcher for the "simple baseline" rather than introducing external heavy machine-learning dependencies like `scikit-learn` (TF-IDF + Logistic Regression). This kept the environment lightweight.

### 9. Zero-Shot Escalation Routing
Decided to use zero-shot JSON prompting for escalation decisions rather than fine-tuning a small classifier. We relied on the large parameter model's inherent reasoning to detect sensitive keywords (like refunds or account access).

### 10. Accepting Conservative Escalation
Upon evaluating the escalation logic, we discovered a precision of 1.0 but a recall of ~0.02. We decided to document this as a core failure mode in `failure_analysis.md` rather than hacking the prompt with aggressive weights, preserving the integrity of the baseline measurement.

### 11. Pure Python Retrieval Index
Implemented the retrieval index using pure Pandas and Jaccard similarity (token overlap). This avoided the need to install vector databases (like FAISS) or embedding models (like SentenceTransformers) in a lightweight environment, strictly adhering to the "keep it simple" mandate.

### 12. Retrieval Context Cap (K=2)
Capped the retrieval context fed to the reply generator at the top $K=2$ historical examples. Providing more examples increased the risk of prompt distraction (the LLM copying the wrong resolution) and increased token costs unnecessarily.

### 13. LLM-as-a-Judge
Automated the end-to-end evaluation using an LLM-as-a-Judge. This allowed us to score generated replies (1-5 scale) against the ideal replies instantly, establishing a continuous quantitative feedback loop without human annotator bottlenecks.

### 14. Robust JSON Extraction
Instead of relying purely on the LLM to output valid JSON, we implemented a robust slicing strategy (`response[response.find("{{"):response.rfind("}}")+1]`). This effectively stripped out markdown block hallucinations (e.g. ```json ... ```) that would otherwise crash the dataframes.

### 15. Qualitative Failure Taxonomy
Decided to categorize failures into 5 distinct qualitative modes (e.g., Generic Apology Loop, Retrieval Distraction) rather than just looking at the numerical 2.99 average score. This provided actionable, human-readable paths for future prompt engineering.
