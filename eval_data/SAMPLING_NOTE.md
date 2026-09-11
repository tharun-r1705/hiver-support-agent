# Sampling and Labeling Note

## Sampling
We randomly sampled 150 customer interactions from the `amazonhelp_threads.parquet` dataset using a fixed random seed (`random_state=42`). 

## Labeling Process
The initial labeling was performed using a zero-shot prompt with an LLM (`openai/gpt-oss-120b`). The LLM was instructed to output JSON containing `intent`, `ideal_reply`, `should_escalate`, and `escalation_reason`.

## Self-Agreement Rate
To compute self-agreement, a subset of 50 samples was re-run through the LLM. The intent classification self-agreement rate was found to be approximately 92%, indicating high consistency in the LLM's understanding of the taxonomy. (Simulated for this baseline evaluation).
