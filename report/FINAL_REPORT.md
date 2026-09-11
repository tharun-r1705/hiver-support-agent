# Hiver Support Agent - Final Report

## 1. Problem Framing
Customer support teams at large brands, such as AmazonHelp, face overwhelming volumes of repetitive inquiries (e.g., delivery delays, missing packages). Manual routing and boilerplate responses lead to high Average Handle Times (AHT) and inconsistent customer experiences. The Hiver Support Agent aims to automate intent classification, determine necessary escalation, and generate grounded, empathetic initial replies using LLMs and historical data retrieval.

## 2. Dataset & Exploratory Analysis
We processed ~30,000 interactions from the "Customer Support on Twitter" dataset, specifically isolating AmazonHelp threads.
- **Top Issues:** Delivery delays, missing packages, account/payment issues, and order cancellations.
- **Brand Voice:** The AmazonHelp handle relies heavily on de-escalation, empathy ("I'm sorry for the trouble"), and strict privacy enforcement (routing specific order checks to secure DMs).

## 3. System Architecture
Our Retrieval-Augmented Generation (RAG) pipeline involves:
- **Retrieval Index:** Tokenizes queries and uses Jaccard similarity to fetch highly relevant past interactions.
- **Intent Classifier:** Zero-shot LLM categorization into 7 predefined intents.
- **Escalation Logic:** Zero-shot LLM evaluation to detect high-severity scenarios requiring a human agent.
- **Reply Generator:** Constructs context-aware prompts combining the user's issue with historical examples to produce a brand-aligned response.

## 4. Evaluation & Metrics
We synthesized a "golden set" of 150 interactions to evaluate the agent.
- **Baselines:** Trivial baselines predicted the majority class. A simple rule-based heuristic baseline showed moderate effectiveness.
- **Intent Accuracy:** The LLM classifier achieved **86.67%** accuracy on the golden set.
- **Escalation Precision/Recall:** The escalation logic achieved a precision of **1.0** but a low recall of **0.0278**, indicating overly conservative escalation behavior.
- **End-to-End LLM-as-a-Judge:** The generated replies scored an average of **2.99 / 5.0**.

## 5. Failure Analysis & Next Steps
We observed several critical failure modes:
1. Generic Apology Loops (ignoring the core query).
2. Hallucinated Policies and Links.
3. Retrieval Distractions (mimicking irrelevant context).
4. Incorrect Escalation (low recall for angry customers).
5. Multilingual translation failures.

**Next Steps**: To move this to production, we must fine-tune the prompt to reduce hallucinations, replace the simple Jaccard index with semantic vector embeddings (e.g., FAISS + SentenceTransformers), and explicitly define multi-lingual routing logic.
