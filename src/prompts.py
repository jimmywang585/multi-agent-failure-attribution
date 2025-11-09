STEP_RATING_SYSTEM_PROMPT = "You are a precise judge in a multi-agent failure analysis task.\n" \
"Assign a NUMBER in [0,100] for each step indicating how likely it is the decisive error.\n" \
"Return strictly the JSON object: {\"scores\": [n0, n1, ...]} with integers."

def build_step_rating_prompt(query, steps):
    numbered = "\n".join([f"[{i}] Agent={s['agent']}: {s['text']}" for i,s in enumerate(steps)])
    return f"""{STEP_RATING_SYSTEM_PROMPT}

Problem:
{query}

Steps:
{numbered}

Now output strictly the JSON object.
"""