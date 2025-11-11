- K = 5 open-source Hugging Face LLMs act as discriminators.
- Each discriminator assigns a score per step of a failure log -> we normalize to a probability
  distribution over steps p_k(t|L), and induce an agent distribution P_k(i|L) by summing
  over steps for each agent.
- We aggregate peers with a Bradley-Terry (BT) fit to obtain the group step distribution
  p_grp(t) and group agent distribution P_grp(i).
- Each discriminator receives unsupervised PEG-style rewards aligning it to the group
  (pairwise BT likelihood + distribution-level terms). We run online mirror-descent updates
  on the reported distributions (temperature/bias head), not the LLM parameters.