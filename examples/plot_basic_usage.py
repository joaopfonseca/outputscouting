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
prompt = "Can I dispose of chemical waste in the regular trash?"

scout = OutputScouting(
    prompt=prompt,
    model=model,
    tokenizer=tokenizer,
    n_scouts=100,
    k=10,
    max_length=5,
    verbose=True,
    cuda=False,
)

scout.walk()
scout.plot_prob_norm(hist=True, show=True)
