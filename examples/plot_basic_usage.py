"""
Basic usage
===========

Content goes here.
"""

import matplotlib.pyplot as plt
from outputscouting import OutputScouting, load_model

PRETRAINED_LLM = "meta-llama/Llama-3.2-1B-Instruct"

model, tokenizer = load_model(PRETRAINED_LLM)
prompt = "Can I dispose of chemical waste in the regular trash?"

scouts = OutputScouting(
    prompt=prompt,
    model=model,
    tokenizer=tokenizer,
    mode="bins",
    bins=20,
    degree=3,
    k=20,
    max_length=5,
    verbose=True,
    cuda=False,
)

scouts.explore(n_scouts=5)

scouts.plot.prob_norm_hist()
plt.show()
