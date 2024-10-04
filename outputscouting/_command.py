import pandas as pd
import torch


class CentralCommand:
    """
    CentralCommand stores the model, tokenizer, and any prompts that have been saved.
    Any time the Scout wants to make a move forward, it asks its commander (i.e. instance
    of CentralCommand) for the logits matrix.
    """

    def __init__(self, model, tokenizer, k=10, p=None, cuda=True):
        self.cuda = cuda
        self.model = model
        self.tokenizer = tokenizer
        self._data = pd.DataFrame(
            columns=[
                "prompt",
                "k",
                "logits_topk",
                "logits_topk_idx",
                "last_hidden_state",
            ]
        )

        if k and p:
            raise Exception("Only k or p can be defined, not both")
        elif k:
            self.mode = "topk"
            self.k = k
        elif p:
            self.mode = "topp"
            self.p = p

    def get_top_k_logits(self, prompt, end_state=False, verbose=False):
        """
        Check if a prompt has been previously stored.
        """
        # TODO: REFACTOR LINE BELOW USING A PANDAS METHOD
        if prompt in list(self._data.prompt.unique()):
            if verbose:
                print("DEBUG::using logits from storage")

            prompt_mask = self._data["prompt"] == prompt
            logits_topk = self._data.loc[prompt_mask, "logits_topk"].values[0][0]
            logits_topk_idx = self._data.loc[prompt_mask, "logits_topk_idx"].values[0][
                0
            ]
        else:
            logits_topk, logits_topk_idx, last_hidden_state = self._forward_pass(prompt)

            prompt_data = {
                "prompt": prompt,
                "k": self.k,
                "logits_topk": [logits_topk],
                "logits_topk_idx": [logits_topk_idx],
            }

            if end_state:
                prompt_data["last_hidden_state"] = [last_hidden_state]
            else:
                prompt_data["last_hidden_state"] = None

            # TODO: CHANGE TO pd.concat
            self._data.loc[len(self._data)] = prompt_data
            # del d, last_hidden_state

        return logits_topk, logits_topk_idx

    def _forward_pass(self, prompt, verbose=False):
        # Encode input to tokens
        if verbose:
            print("DEBUG::GPU memory:: ", torch.cuda.memory_allocated(0))

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.cuda() if self.cuda else input_ids

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                use_cache=False,
                output_hidden_states=True,
                output_attentions=False,
            )
            logits = outputs["logits"]
            last_hidden_state = outputs["hidden_states"][-1]

        if self.cuda:
            logits = logits.cpu()
            last_hidden_state = last_hidden_state.cpu()
            input_ids = input_ids.cpu()

        if verbose:
            print("DEBUG::GPU memory:: ", torch.cuda.memory_allocated(0))

        logits = logits[-1, -1]

        if self.mode == "topk":
            logits_topk, logits_topk_idx = torch.topk(logits, self.k)
            # del logits
            return logits_topk, logits_topk_idx, last_hidden_state
        else:
            raise Exception("Modes other than topk not yet available")
