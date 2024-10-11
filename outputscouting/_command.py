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
                "output",
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

    def get_top_k_logits(self, all_ids, output, end_state=False, verbose=False):
        """
        Check if a prompt has been previously stored.
        """
        if output in list(self._data.output.unique()):
            if verbose:
                print("DEBUG::using logits from storage")

            output_mask = self._data["output"] == output
            logits_topk = self._data.loc[output_mask, "logits_topk"].values[0][0]
            logits_topk_idx = self._data.loc[output_mask, "logits_topk_idx"].values[0][
                0
            ]
        else:
            logits_topk, logits_topk_idx, last_hidden_state = self._forward_pass(
                all_ids
            )

            output_data = {
                "output": output,
                "k": self.k,
                "logits_topk": [logits_topk],
                "logits_topk_idx": [logits_topk_idx],
            }

            if end_state:
                output_data["last_hidden_state"] = [last_hidden_state]
            else:
                output_data["last_hidden_state"] = None

            # TODO: CHANGE TO pd.concat
            self._data.loc[len(self._data)] = output_data
            # del d, last_hidden_state

        return logits_topk, logits_topk_idx

    def _forward_pass(self, all_ids, verbose=False):
        # Encode input to tokens
        if verbose:
            print("DEBUG::GPU memory:: ", torch.cuda.memory_allocated(0))

        all_ids = all_ids.cuda() if self.cuda else all_ids

        with torch.no_grad():
            outputs = self.model(
                all_ids,
                use_cache=False,
                output_hidden_states=True,
                output_attentions=False,
            )
            logits = outputs["logits"]
            last_hidden_state = outputs["hidden_states"][-1]

        if self.cuda:
            logits = logits.cpu()
            last_hidden_state = last_hidden_state.cpu()
            all_ids = all_ids.cpu()

        if verbose:
            print("DEBUG::GPU memory:: ", torch.cuda.memory_allocated(0))

        logits = logits[-1, -1]

        if self.mode == "topk":
            logits_topk, logits_topk_idx = torch.topk(logits, self.k)
            # del logits
            return logits_topk, logits_topk_idx, last_hidden_state
        else:
            raise Exception("Modes other than topk not yet available")
