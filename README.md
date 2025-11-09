# Failure Attribution via PEG + Bradley-Terry (Who&When)

Implements a multi-discriminator, training-free failure attribution system for Who&When-style data:

- K = 5 open-source Hugging Face LLMs act as discriminators.
- Each discriminator assigns a score per step of a failure log -> we normalize to a probability
  distribution over steps p_k(t|L), and induce an agent distribution P_k(i|L) by summing
  over steps for each agent.
- We aggregate peers with a Bradley-Terry (BT) fit to obtain the group step distribution
  p_grp(t) and group agent distribution P_grp(i).
- Each discriminator receives unsupervised PEG-style rewards aligning it to the group
  (pairwise BT likelihood + distribution-level terms). We run online mirror-descent updates
  on the reported distributions (temperature/bias head), not the LLM parameters.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.main --config config.yaml --data examples/sample_log.json --epochs 3
```

Real Who&When JSONL format (one sample per line) example:
{
  "log_id": "abc123",
  "query": "problem text",
  "steps": [{"agent":"A","text":"..."}, {"agent":"B","text":"..."}],
  "gold_agent": "B",
  "gold_step_index": 4
}
