# What's Misleading About My Headline Number

The end-to-end evaluation yielded an average LLM-as-a-Judge score of 2.99 / 5 and an intent classification accuracy of 86.67%. However, these "headline numbers" can be highly misleading. Based on 6 standard rigor checks, here are the underlying issues:

### 1. Data Contamination & Synthetic Golden Set
Our "golden set" was synthetically generated using the exact same LLM (`gpt-oss-120b`) that we are evaluating. High intent accuracy might just mean the model is highly consistent with its own zero-shot outputs, rather than matching true human-curated ground truth.

### 2. LLM-as-a-Judge Bias (Self-Preference)
The evaluation relied on the LLM to judge its own generated replies. LLMs are notorious for preferring their own linguistic style (self-preference bias), meaning the score may be artificially inflated or deflated compared to true human evaluation.

### 3. Class Imbalance Obscuring Critical Failures
While the 86.67% intent accuracy looks good, it hides performance on minority but highly critical classes (like `MISSING_PACKAGE`). A model that defaults to the majority class can achieve high overall accuracy while failing catastrophically on the most important edge cases.

### 4. Poor Recall on Escalation
Our escalation precision was 1.0, but recall was only ~2.7%. The headline intent accuracy completely ignores the fact that the agent systematically fails to identify angry customers who desperately need human intervention, which is a major business risk.

### 5. Prompt Sensitivity & Brittleness
The headline score is a single snapshot. We haven't tested how sensitive the LLM is to minor prompt variations. A slight rephrasing of the system prompt could drastically alter the 2.99 average score, meaning the metric is likely not stable.

### 6. Metric Misalignment with Real-World Objectives
A 3/5 score on a synthetic prompt rubric doesn't guarantee real-world customer satisfaction (CSAT) or reduced average handle time (AHT). The metric lacks proven correlation with the actual business goals of the support team.
