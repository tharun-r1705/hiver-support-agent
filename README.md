# Hiver Support Agent

An end-to-end LLM-based customer support agent built to automate intent classification, escalation detection, and context-aware reply generation for the AmazonHelp Twitter support dataset.

## Setup Instructions

1. Ensure you have Python installed.
2. Activate the virtual environment:
   ```powershell
   myvenv\Scripts\activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Set up your `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_key_here
   ```

## Running the Pipeline

Ensure that you add the project root to your `PYTHONPATH` before running scripts. For PowerShell:
```powershell
$env:PYTHONPATH="."
```

**1. Generate the Golden Set**
Samples 150 interactions and automatically labels intent, ideal replies, and escalation using the LLM:
```powershell
python src/eval/generate_golden_set.py
```

**2. Run Baselines**
Calculate simple heuristic and trivial baselines on the golden set:
```powershell
python src/eval/baseline_trivial.py
python src/eval/baseline_simple.py
```

**3. Evaluate Intent Classification**
Evaluates zero-shot LLM intent classification accuracy:
```powershell
python src/intents/classifier.py
```

**4. Evaluate Escalation Logic**
Calculates precision and recall for the LLM escalation router:
```powershell
python src/escalation/decision.py
```

**5. Generate Grounded Replies**
Uses a token-overlap Jaccard retrieval index to find historical context and generate brand-aligned replies:
```powershell
python src/reply/generator.py
```

**6. Run LLM-as-a-Judge Evaluation**
Evaluates the quality of generated replies on a 1-5 scale:
```powershell
python src/eval/run_eval.py
```

## Documentation
Please check the `report/` folder for the full System Design, Final Report, Decision Log, and Failure Analysis.
