import numpy as np
from ._scout import Scout


class ExtensiveSearch(Scout):

    def __init__(self, prompt, command=None, k=None, p=None, max_length=np.inf):
        super().__init__(
            prompt=prompt, command=command, k=k, p=p, max_length=max_length
        )
        self.init_prompt_len = len(self.command.tokenizer.tokenize(prompt))

    def _step(self, prompt, verbose=False):

        curr_len = len(self.command.tokenizer.tokenize(prompt)) - self.init_prompt_len
        if (
            prompt.find(self.command.tokenizer.eos_token) == -1
            and curr_len < self.max_length
        ):
            probs = self.command.get_probs(prompt, verbose=verbose)
            if self.mode == "topk":
                probs_top, texts = self._get_top_k(probs)
            elif self.mode == "topp":
                probs_top, texts = self._get_top_p(probs)

            for text in texts:
                prompt_ = prompt + text
                if verbose:
                    print("DEBUG2::", prompt_)
                self._step(prompt_, verbose=verbose)

    def explore(self, verbose=False):
        self._step(self.init_prompt, verbose=verbose)
        return self
