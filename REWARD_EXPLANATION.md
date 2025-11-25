        # How Rewards Are Actually Calculated

## Key Point: Rewards are PER DISCRIMINATOR, not for the consensus

Each of the 5 discriminators gets its own reward based on:
1. How correct **that specific discriminator** is (supervised)
2. How much **that specific discriminator** agrees with the BT consensus (unsupervised)

## Supervised Reward (When Ground Truth Available)

For each discriminator k:
```
R_sup = log(p_k(correct_step)) + log(P_k(correct_agent))
```

**What this means:**
- We look at discriminator k's OWN probability distribution
- We check: what probability did discriminator k assign to the CORRECT step?
- We check: what probability did discriminator k assign to the CORRECT agent?
- We take the log of those probabilities and add them up

**Example:**
- Ground truth: step 3 is wrong, agent "Alice" is responsible
- Discriminator 1 says: step 3 has 80% probability, Alice has 70% probability
- Reward = log(0.80) + log(0.70) = -0.223 - 0.357 = -0.58 (good reward, close to 0)

- Discriminator 2 says: step 5 has 90% probability, Bob has 80% probability  
- Reward = log(0.0) + log(0.0) = -100 - 100 = -200 (very bad reward)

**Important:** This is NOT about whether the consensus is correct. It's about whether THIS SPECIFIC DISCRIMINATOR is correct.

## Unsupervised Reward (BT Consensus Alignment)

For each discriminator k:
```
R_unsup = alignment between discriminator k's distribution and BT consensus distribution
```

**What this means:**
- First, we compute the BT consensus from ALL 5 discriminators
- Then, for each discriminator k, we measure how similar k's distribution is to the consensus
- Discriminators that agree with the consensus get rewarded
- Discriminators that disagree get penalized

**Components:**
1. **Pairwise alignment**: Does discriminator k's pairwise preferences match the BT consensus preferences?
2. **Step distributional alignment**: Does discriminator k's step distribution match the consensus step distribution? (measured by cross-entropy)
3. **Agent distributional alignment**: Does discriminator k's agent distribution match the consensus agent distribution?

**Example:**
- BT consensus says: step 3 has 60% probability, step 4 has 30%, step 5 has 10%
- Discriminator 1 says: step 3 has 70%, step 4 has 20%, step 5 has 10% (pretty similar)
- Discriminator 2 says: step 5 has 90%, step 3 has 5%, step 4 has 5% (very different)
- Discriminator 1 gets a better unsupervised reward (agrees more with consensus)
- Discriminator 2 gets a worse unsupervised reward (disagrees with consensus)

**Important:** This is NOT about whether the consensus is correct. It's about whether THIS SPECIFIC DISCRIMINATOR agrees with the group.

## Total Reward

For each discriminator k:
```
If labels available:
  R_total = R_sup + λ * R_unsup
  
If no labels:
  R_total = R_unsup
```

## What We're NOT Doing

We are **NOT** rewarding based on:
- Whether the BT consensus is correct
- Whether the consensus gives high probability to ground truth

We ARE rewarding:
- Each individual discriminator's correctness (supervised)
- Each individual discriminator's agreement with consensus (unsupervised)

## Why This Design?

The idea is:
1. Supervised reward encourages each discriminator to be individually correct
2. Unsupervised reward encourages discriminators to converge toward a consensus
3. Together, this should make discriminators both accurate AND consistent

The consensus emerges from the discriminators, but we don't directly reward "is the consensus right" - we reward "is each discriminator right" and "does each discriminator agree with others".




