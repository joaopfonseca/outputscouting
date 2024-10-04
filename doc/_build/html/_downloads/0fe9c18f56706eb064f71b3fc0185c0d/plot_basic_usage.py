"""
Basic usage
===========

Content goes here.
"""

import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from outputscouting import OutputScouting

PRETRAINED_LLM = "TinyLlama/TinyLlama-1.1B-step-50K-105b"

model = AutoModelForCausalLM.from_pretrained(
    PRETRAINED_LLM,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_LLM, padding_side="left")
prompt = "Can I dispose of chemical waste in the regular trash? Answer:"

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
