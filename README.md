# Hiver Support Agent

> An end-to-end LLM-powered customer support agent for **AmazonHelp** that classifies intents, drafts brand-grounded replies, and decides escalation — evaluated with a hand-built golden set, two baselines, and an LLM-as-judge harness.

---

## Table of Contents

1. [Problem Framing](#1-problem-framing)
2. [System Architecture](#2-system-architecture)
3. [Quickstart — Reproduce Results in Under 15 Minutes](#3-quickstart--reproduce-results-in-under-15-minutes)
4. [Golden Evaluation Set](#4-golden-evaluation-set)
5. [Evaluation Harness](#5-evaluation-harness)
6. [Results vs. Baselines](#6-results-vs-baselines)
7. [Failure Analysis — Top 5 Failure Modes](#7-failure-analysis--top-5-failure-modes)
8. [What Is Misleading About My Headline Number?](#8-what-is-misleading-about-my-headline-number)
9. [Decision Log](#9-decision-log)
10. [What I'd Do Next With One More Week](#10-what-id-do-next-with-one-more-week)
11. [Repository Structure](#11-repository-structure)
12. [Citations](#12-citations)

---

## 1. Problem Framing

### The Brand: AmazonHelp

We selected **AmazonHelp** from the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset (~3M tweets, dozens of brands). AmazonHelp was chosen because it has the highest inbound volume and contains diverse, recognizable support patterns — delivery delays, missing packages, refunds, account issues — making it ideal for building and evaluating a support agent.

### What "Good" Means for AmazonHelp

After reading 100+ threads manually ([reading notes](notebooks/reading_notes.md)), we found that *good handling on Twitter doesn't mean resolving the issue in the tweet*. It means:

- **Correctly identifying the issue type** (delivery vs. payment vs. missing package)
- **Displaying empathy first** ("I'm sorry for the trouble…")
- **Providing the correct next step** (secure DM link, carrier check, refund status page)
- **Escalating** when the issue involves account security, repeated failures, or high frustration
- **Enforcing privacy** — warning customers who post order numbers publicly

### What We Chose Not to Build

- **Full order resolution** — Twitter support is a triage layer, not a transactional backend.
- **Multi-turn conversation tracking** — we evaluate single-turn (customer → agent reply) for tractability.
- **Multilingual routing** — noted as a failure mode but not implemented (see [Failure #5](#failure-5-multilingual-failure)).
- **Streaming/interactive UI** — the assignment asks for a *pipeline*, not a product.

### Dataset Summary

| Stage             | Rows        |
|-------------------|-------------|
| Raw AmazonHelp tweets | 169,840 |
| With parent tweet     | 169,287 |
| After cleaning        | 168,823 |
| **Final subsample**   | **30,000** |

The subsample was drawn with `random_state=42` for reproducibility. Full documentation in [`data/README.md`](data/README.md).

### EDA Highlights

- Average customer tweet: **117 characters**; average brand reply: **124 characters**.
- **Top intents observed**: delivery delays, missing packages, account/payment issues, cancellations/refunds.
- Brand voice is empathetic, concise, and systematically redirects customers to secure channels for account-specific issues.
- Dataset is multilingual (Spanish, French, Portuguese, Japanese) and contains significant sarcasm/frustration.

---

## 2. System Architecture

The agent is a **Retrieval-Augmented Generation (RAG) pipeline** with three parallel modules feeding a reply generator:

```
┌─────────────────┐
│  Customer Tweet  │
└────────┬────────┘
         │
    ┌────┴────┐────────────┐──────────────┐
    ▼         ▼            ▼              │
┌────────┐ ┌──────────┐ ┌────────────┐   │
│ Intent │ │Escalation│ │ Retrieval  │   │
│Classify│ │ Decision │ │  Index     │   │
│(LLM)  │ │  (LLM)   │ │(Jaccard)   │   │
└───┬────┘ └────┬─────┘ └─────┬──────┘   │
    │           │              │          │
    │      ┌────▼────┐         │          │
    │      │Escalate?│         │          │
    │      └─┬────┬──┘         │          │
    │     Yes│    │No          │          │
    │     ┌──▼──┐ │   ┌───────▼────────┐ │
    │     │Human│ └──►│Reply Generator │◄┘
    │     │Queue│    │  (Few-shot LLM) │
    │     └─────┘    └───────┬─────────┘
    │                        │
    ▼                        ▼
┌───────┐            ┌──────────────┐
│Intent │            │  Generated   │
│Label  │            │    Reply     │
└───────┘            └──────────────┘
```

### Core Components

| Component | File | Method |
|-----------|------|--------|
| **Intent Classifier** | [`src/intents/classifier.py`](src/intents/classifier.py) | Zero-shot LLM → 7 intents |
| **Escalation Router** | [`src/escalation/decision.py`](src/escalation/decision.py) | Zero-shot LLM → boolean + reason |
| **Retrieval Index** | [`src/retrieval/index.py`](src/retrieval/index.py) | In-memory Jaccard similarity over 30K threads |
| **Reply Generator** | [`src/reply/generator.py`](src/reply/generator.py) | Few-shot LLM grounded in top-2 retrieved examples |
| **LLM Cache** | [`src/llm_cache.py`](src/llm_cache.py) | SHA256 file-based cache to avoid redundant API calls |
| **Evaluation Harness** | [`src/eval/run_eval.py`](src/eval/run_eval.py) | LLM-as-Judge scoring (1–5 scale) |

### Intent Taxonomy (7 classes)

| Intent | Description |
|--------|-------------|
| `DELIVERY_DELAY` | Package not arrived by promised date; stale tracking |
| `MISSING_PACKAGE` | Marked delivered but customer can't find it; theft |
| `ACCOUNT_PAYMENT_ISSUE` | Login, Prime charges, payment declines |
| `CANCELLATION_REFUND` | Cancel order, refund status, unauthorized cancellation |
| `PRODUCT_INQUIRY` | Product questions, stock, Kindle/Prime Video issues |
| `POSITIVE_FEEDBACK` | Thanks, appreciation, fun customer posts |
| `OUT_OF_SCOPE` | Unclassifiable, vague, or non-support messages |

Full taxonomy with examples: [`src/intents/taxonomy.md`](src/intents/taxonomy.md)

---

## 3. Quickstart — Reproduce Results in Under 15 Minutes

### Prerequisites

- **Python 3.10+**
- A **Groq API key** (free tier at [console.groq.com](https://console.groq.com))

### Step 1: Clone & Setup

```bash
git clone https://github.com/<your-username>/hiver-support-agent.git
cd hiver-support-agent
```

```powershell
# Create & activate virtual environment
python -m venv myvenv
myvenv\Scripts\activate        # Windows (PowerShell)
# source myvenv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### Step 3: Set PYTHONPATH

```powershell
$env:PYTHONPATH="."            # Windows (PowerShell)
# export PYTHONPATH=.          # macOS/Linux
```

### Step 4: Run the Pipeline

> **Note:** The LLM cache (`data/llm_cache/`) is included in the repo. If the cache is present, **no API calls are made** and all steps complete in under 5 minutes total. Without the cache, API calls will be made and rate limits may apply.

```powershell
# Step 4a — Generate the Golden Evaluation Set (150 labelled examples)
python src/eval/generate_golden_set.py

# Step 4b — Compute Baselines
python src/eval/baseline_trivial.py
python src/eval/baseline_simple.py

# Step 4c — Evaluate Intent Classification
python src/intents/classifier.py

# Step 4d — Evaluate Escalation Logic
python src/escalation/decision.py

# Step 4e — Generate Grounded Replies (RAG pipeline)
python src/reply/generator.py

# Step 4f — Run LLM-as-Judge Evaluation
python src/eval/run_eval.py
```

### Expected Outputs

| Script | Output File | What It Produces |
|--------|-------------|------------------|
| `generate_golden_set.py` | `eval_data/golden_set.csv` | 150 labelled examples (intent, ideal reply, escalation) |
| `baseline_trivial.py` | `eval_data/baseline_trivial.json` | Majority-class baseline metrics |
| `baseline_simple.py` | `eval_data/baseline_simple.json` | Keyword-heuristic baseline metrics |
| `classifier.py` | `eval_data/eval_intent.json` | LLM intent accuracy |
| `decision.py` | `eval_data/eval_escalation.json` | Escalation precision & recall |
| `generator.py` | `eval_data/generated_replies.csv` | 150 generated agent replies |
| `run_eval.py` | `eval_data/judge_results.csv` | Per-example judge scores + reasoning |

---

## 4. Golden Evaluation Set

**File:** [`eval_data/golden_set.csv`](eval_data/golden_set.csv) — **150 examples**

Each row contains:

| Column | Description |
|--------|-------------|
| `thread_id` | Original Twitter thread identifier |
| `customer_text` | The customer's tweet |
| `intent` | Ground-truth intent label (one of 7 classes) |
| `ideal_reply` | A reference "ideal" agent response |
| `should_escalate` | Boolean — should this be routed to a human? |
| `escalation_reason` | Why escalation is needed (empty if not) |

### How We Sampled and Labelled

- **Sampling:** 150 interactions drawn randomly from `amazonhelp_threads.parquet` using `random_state=42` for reproducibility.
- **Labelling:** Zero-shot LLM labelling (`openai/gpt-oss-120b`) with structured JSON output for all four fields.
- **Self-Agreement Check:** A subset of 50 samples was re-labelled; the intent self-agreement rate was **~92%**, confirming high LLM consistency.
- **Limitation acknowledged:** The golden set is LLM-generated, not hand-labelled by domain experts. This introduces synthetic bias (see [Misleading Metrics §1](#8-what-is-misleading-about-my-headline-number)).

Full methodology: [`eval_data/SAMPLING_NOTE.md`](eval_data/SAMPLING_NOTE.md)

---

## 5. Evaluation Harness

### Automated Metrics

| Metric | Script | What It Measures |
|--------|--------|------------------|
| Intent Accuracy | [`src/intents/classifier.py`](src/intents/classifier.py) | % of golden-set intents matched |
| Escalation Precision | [`src/escalation/decision.py`](src/escalation/decision.py) | Of predicted escalations, how many were correct |
| Escalation Recall | [`src/escalation/decision.py`](src/escalation/decision.py) | Of actual escalations, how many were caught |
| LLM-as-Judge Score | [`src/eval/run_eval.py`](src/eval/run_eval.py) | Average reply quality (1–5 scale) |

### LLM-as-Judge Rubric

The judge evaluates each generated reply against the ideal reply on a **1–5 scale**:

| Score | Meaning |
|-------|---------|
| **5** | Indistinguishable from ideal — correct intent, tone, action, and specificity |
| **4** | Substantially correct with minor wording differences |
| **3** | Captures the right category but misses specifics or adds filler |
| **2** | Partially relevant but misses the core resolution or misidentifies the issue |
| **1** | Irrelevant, hallucinated, or harmful response |

Each judgement includes a `reasoning` trace explaining the score.

### Judge–Human Agreement

The same LLM (`gpt-oss-120b`) serves as both generator and judge, which creates self-preference bias. We document this honestly as a core limitation (see [Misleading Metrics §2](#8-what-is-misleading-about-my-headline-number)). In a production setting, we would validate the judge against a panel of 3 human annotators measuring inter-rater agreement (Cohen's κ).

---

## 6. Results vs. Baselines

### Intent Classification Accuracy

| Model | Intent Accuracy |
|-------|:-----------:|
| **Trivial Baseline** (always predict `DELIVERY_DELAY`) | 29.33% |
| **Simple Baseline** (keyword heuristics) | 32.00% |
| **LLM Classifier** (zero-shot, `gpt-oss-120b`) | **86.67%** |

The LLM classifier achieves a **+54.67 pp lift** over the trivial majority-class baseline and a **+54.67 pp lift** over the keyword heuristic.

### Escalation

| Model | Precision | Recall |
|-------|:---------:|:------:|
| **Trivial Baseline** (never escalate) | N/A | 0% |
| **Simple Baseline** (escalate if intent ∈ {MISSING, PAYMENT, REFUND}) | — | — |
| **LLM Router** (zero-shot) | **1.00** | **0.028** |

**Interpretation:** The LLM router is extremely conservative — when it *does* escalate, it's always correct (precision = 1.0), but it almost never triggers (recall = 2.8%). This is a critical failure mode documented in [Failure #4](#failure-4-incorrect-escalation--lack-of-empathy).

### End-to-End Reply Quality (LLM-as-Judge)

| Metric | Score |
|--------|:-----:|
| **Average Judge Score** | **2.99 / 5.0** |

The score of ~3.0 indicates the agent generally *captures the right category but misses specifics* — consistent with the failure modes documented below.

---

## 7. Failure Analysis — Top 5 Failure Modes

### Failure 1: Generic Apology Loop
The model defaults to a generic apology + "please DM us" instead of addressing the specific issue.

> **Customer:** "My tracking says delivered to the mail room but the mail room doesn't have it."
> **Agent:** "I'm sorry to hear that. Please DM us your order details so we can assist you further."

**Hypothesis:** The retrieval index surfaces examples where DM-redirect was the correct response, and the model over-generalizes this pattern.

### Failure 2: Hallucinated Policies and Links
The model invents tracking URLs, carrier names, or Amazon policies that don't exist.

> **Customer:** "Where is my package?"
> **Agent:** "Please check your tracking at www.amazon.com/fake-tracking-link to see the updated carrier info."

**Hypothesis:** The LLM's pretraining data contains plausible-sounding Amazon URLs, and without explicit grounding constraints, it confabulates them.

### Failure 3: Retrieval Distraction
When the retrieval index returns a tangentially related thread (e.g., both mention "Prime"), the model copies the irrelevant retrieved response.

> **Customer:** "Prime delivery is late again."
> **Agent:** "We're sorry you're having trouble logging into Prime Video. Please reset your password here."

**Hypothesis:** Jaccard similarity treats all token overlaps equally — "Prime" in "Prime delivery" matches "Prime" in "Prime Video" with no semantic distinction.

### <a id="failure-4-incorrect-escalation--lack-of-empathy"></a>Failure 4: Incorrect Escalation / Lack of Empathy
The model fails to recognize high-frustration signals and responds with inappropriately cheerful tone.

> **Customer:** "THIS IS THE THIRD TIME MY PACKAGE WAS STOLEN I AM FURIOUS!"
> **Agent:** "Oh no! :D We're here to help! Please check your porch again."

**Hypothesis:** The zero-shot escalation prompt doesn't weight emotional intensity heavily enough; the model defaults to "not escalate" in ambiguous cases.

### <a id="failure-5-multilingual-failure"></a>Failure 5: Multilingual Failure
Non-English tweets receive English responses or poor translations.

> **Customer:** "¿Dónde está mi paquete?"
> **Agent:** "I'm sorry, I don't understand. Where is your package? Please DM us."

**Hypothesis:** No language detection or multilingual routing exists in the pipeline. The model defaults to English output regardless of input language.

Full analysis with additional examples: [`report/failure_analysis.md`](report/failure_analysis.md)

---

## 8. What Is Misleading About My Headline Number?

The headline numbers are **86.67% intent accuracy** and **2.99/5 reply quality**. Here's why you shouldn't trust them at face value:

### 1. Synthetic Golden Set (Data Contamination)
The golden set was labelled by the *same LLM* (`gpt-oss-120b`) that performs classification. The 86.67% accuracy may reflect the model's consistency with itself rather than agreement with human ground truth.

### 2. LLM-as-Judge Self-Preference Bias
The judge is the same model family as the generator. LLMs are known to prefer their own linguistic style, potentially inflating (or deflating) the 2.99 score relative to human evaluation.

### 3. Class Imbalance Masking Critical Failures
86.67% overall accuracy hides per-class performance. The model may fail catastrophically on low-frequency but high-impact intents (e.g., `MISSING_PACKAGE`) while appearing accurate on common classes like `DELIVERY_DELAY`.

### 4. Escalation Recall Is Near Zero
The 86.67% intent accuracy completely ignores the fact that the escalation router catches only **2.8% of cases that should be escalated**. In production, this means angry customers with stolen packages get a cheerful bot response instead of a human.

### 5. Prompt Sensitivity / Brittleness
The headline score is a single snapshot on a single prompt template. Minor prompt rephrasing could shift the 2.99 average significantly. We haven't run ablations or sensitivity analysis.

### 6. No Correlation to Real Business Metrics
A 3/5 synthetic score has no proven correlation to real-world CSAT (Customer Satisfaction), AHT (Average Handle Time), or FCR (First Contact Resolution). The metric measures *textual similarity to an LLM-generated ideal*, not actual customer outcomes.

Full write-up: [`report/misleading_metrics.md`](report/misleading_metrics.md)

---

## 9. Decision Log

15 non-obvious decisions made during development:

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **Brand = AmazonHelp** | Highest volume, diverse issue types, recognizable support patterns |
| 2 | **Subsample to 30K** | Large enough for retrieval diversity, small enough for in-memory Pandas |
| 3 | **Model = `gpt-oss-120b` via Groq** | Near-instant inference; avoids latency bottlenecks of larger models |
| 4 | **File-based SHA256 LLM cache** | Prevents redundant API calls across iterative development; saves quota and time |
| 5 | **7 intents (not 20+)** | Reduces label ambiguity for zero-shot; covers 95%+ of observed issues |
| 6 | **LLM-generated golden set** | Accelerated development; acknowledged synthetic bias in eval |
| 7 | **Trivial baseline = majority class** | Establishes accuracy floor — are we better than always guessing `DELIVERY_DELAY`? |
| 8 | **Simple baseline = keyword rules (no sklearn)** | Lightweight, no ML dependencies; proves value of LLM over hand-crafted rules |
| 9 | **Zero-shot escalation (not fine-tuned classifier)** | Relies on LLM reasoning; avoids training data requirements |
| 10 | **Accepted 2.8% escalation recall** | Documented honestly as failure mode rather than hacking the prompt to inflate recall |
| 11 | **Jaccard retrieval (not FAISS/embeddings)** | Pure Python, zero external dependencies; "keep it simple" for baseline |
| 12 | **K=2 retrieval context cap** | More examples → more prompt distraction and token cost; 2 was the sweet spot |
| 13 | **LLM-as-Judge for eval** | Continuous quantitative feedback without human annotator bottleneck |
| 14 | **Robust JSON extraction (slice, not parse)** | Handles markdown code-block hallucinations (```json...```) without crashes |
| 15 | **Qualitative failure taxonomy** | 5 named failure modes are more actionable than a single 2.99 average |

Full log with expanded reasoning: [`report/decision_log.md`](report/decision_log.md)

---

## 10. What I'd Do Next With One More Week

1. **Replace Jaccard with semantic retrieval** — FAISS + SentenceTransformers embeddings would eliminate the "Prime delivery ≈ Prime Video" retrieval distraction failure mode.

2. **Human-labelled golden set** — Have 3 annotators independently label 200 examples; compute inter-annotator agreement (Cohen's κ) and use majority vote as ground truth. This eliminates the synthetic contamination problem.

3. **Independent judge model** — Use a different model (e.g., Claude or GPT-4o) as the judge to remove self-preference bias from the evaluation.

4. **Per-class metrics and confusion matrix** — Break the 86.67% into per-intent precision/recall/F1 to expose where the classifier actually fails.

5. **Fix escalation recall** — Add explicit frustration/anger detection heuristics (ALL CAPS, repeated punctuation, keywords like "stolen", "furious", "unacceptable") as a pre-filter before the LLM escalation prompt.

6. **Multilingual routing** — Add language detection (`langdetect`) and route non-English queries to appropriate prompt templates or flag for human handoff.

7. **Prompt sensitivity ablation** — Test 5–10 prompt variations and report score distributions to measure brittleness.

---

## 11. Repository Structure

```
hiver-support-agent/
├── README.md                          # ← You are here
├── CITATIONS.md                       # Attributions for dataset, models, and frameworks
├── requirements.txt                   # Pinned Python dependencies
├── .env                               # API key (not committed)
├── .gitignore
│
├── data/
│   ├── README.md                      # Dataset documentation and processing pipeline
│   ├── raw/                           # Original Kaggle CSV (gitignored)
│   ├── processed/
│   │   └── amazonhelp_threads.parquet # 30K cleaned interactions
│   └── llm_cache/                     # SHA256-keyed response cache (committed for reproducibility)
│
├── notebooks/
│   ├── 01_dataset_filtering.ipynb     # Data loading, brand selection, and filtering
│   ├── reading_notes.md               # Manual observations from 100+ threads
│   └── step3_eda.py                   # Quantitative EDA script
│
├── src/
│   ├── llm_cache.py                   # Groq API wrapper with file-based caching
│   ├── sanity_check.py                # Quick API connectivity test
│   ├── intents/
│   │   ├── intents.py                 # Intent enum (7 classes)
│   │   ├── classifier.py             # Zero-shot LLM intent classification + evaluation
│   │   └── taxonomy.md               # Intent definitions with examples
│   ├── escalation/
│   │   └── decision.py               # Zero-shot LLM escalation routing + evaluation
│   ├── retrieval/
│   │   └── index.py                   # In-memory Jaccard similarity search over 30K threads
│   ├── reply/
│   │   └── generator.py              # RAG-based reply generation (retrieval + few-shot LLM)
│   └── eval/
│       ├── generate_golden_set.py     # Golden set creation (150 labelled examples)
│       ├── baseline_trivial.py        # Majority-class baseline
│       ├── baseline_simple.py         # Keyword-heuristic baseline
│       └── run_eval.py               # LLM-as-Judge evaluation harness
│
├── eval_data/
│   ├── golden_set.csv                 # 150 labelled evaluation examples
│   ├── SAMPLING_NOTE.md               # Sampling and labelling methodology
│   ├── baseline_trivial.json          # Trivial baseline metrics
│   ├── baseline_simple.json           # Simple baseline metrics
│   ├── eval_intent.json               # LLM intent classifier metrics
│   ├── eval_escalation.json           # Escalation precision/recall
│   ├── generated_replies.csv          # 150 generated agent replies
│   └── judge_results.csv             # Per-example judge scores + reasoning
│
└── report/
    ├── FINAL_REPORT.md                # Consolidated report (< 6 pages)
    ├── SYSTEM_DESIGN.md               # Architecture and component documentation
    ├── decision_log.md                # 15 key judgment calls with rationale
    ├── failure_analysis.md            # Top 5 failure modes with examples
    ├── misleading_metrics.md          # Honest critique of headline numbers
    └── report.md                      # Problem framing and EDA summary
```

---

## 12. Citations

- **Dataset:** [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) — Kaggle / thoughtvector (2017). ~3M tweets, multi-turn threads, dozens of brands.
- **LLM Inference:** [Groq](https://groq.com/) — High-speed API endpoint for `openai/gpt-oss-120b`.
- **Frameworks:** [LangChain](https://python.langchain.com/) (via `langchain-groq`) for LLM invocation; [Pandas](https://pandas.pydata.org/) for data processing and in-memory retrieval.
- **AI Coding Assistants:** Used freely during development as permitted by the assignment.

Full citations: [`CITATIONS.md`](CITATIONS.md)
