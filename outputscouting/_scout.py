import numpy as np
import pandas as pd
import torch


class Scout:
    """
    Explores a single path through the output space/tree.
    """

    def __init__(
        self,
        prompt,
        commander=None,
        t=0.5,
        t_aux=None,
        max_length=np.inf,
        random_state=None,
    ):
        self.prompt = prompt
        self.commander = commander
        self.t = t
        self.t_aux = t_aux
        self.max_length = max_length
        self.random_state = random_state

    def get_data(self):
        out = {}
        out["phrase"] = "".join(self._data["selected_text"].values)

        # Store metadata (temp, t_aux and max_length)
        out["temp"] = self.t
        out["temp_aux"] = self.t_aux
        out["max_length"] = self.max_length

        # Compute overall probability
        probs = self._data.apply(lambda x: x["probs"][x["selected_idx"]], axis=1).values
        out["prob"] = np.prod(probs).item()

        # Compute normalized probabilities
        log_probs = np.log(probs)
        log_prob_sum = np.sum(log_probs)
        log_prob_sum_l = log_prob_sum / len(probs)
        prob_norm = np.exp(log_prob_sum_l)
        out["prob_norm"] = prob_norm.item()

        return out

    def explore(self, verbose=False):
        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(self.random_state)

        if hasattr(self, "_data"):
            raise Exception("Data already exists, cannot explore again.")
        else:
            self._data = pd.DataFrame(
                columns=[
                    "texts",
                    "token_ids",
                    "probs",
                    "selected_idx",
                    "selected_text",
                ]
            )

        t_aux = self.t_aux or self.t
        eos_token = self.commander.tokenizer.eos_token

        prompt = [
            {"role": "user", "content": self.prompt},
            {"role": "assistant", "content": ""},
        ]
        prompt_ids = self.commander.tokenizer.apply_chat_template(
            prompt, return_tensors="pt"
        )
        prompt_ids = prompt_ids[0][:-1].reshape(1, -1)

        output = ""
        output_ids = torch.tensor([], dtype=torch.long)

        if verbose:
            print(f"DEBUG::t={self.t}, t_aux={t_aux}:: ", end="", flush=True)

        end_state = False
        while not end_state:

            # end_state is used to retrieve the last set of logits (inc. eos_token)
            if output.find(eos_token) >= 0 or self._data.shape[0] >= self.max_length:
                end_state = True

            if self.commander.cuda:
                prompt_ids = prompt_ids.cuda()
                output_ids = output_ids.cuda()

            if self.commander.mode == "topk":
                all_ids = torch.cat([prompt_ids, output_ids], dim=-1)
                logits_top, logits_top_idx = self.commander.get_top_k_logits(
                    all_ids, output, end_state=end_state, verbose=False
                )
            else:
                raise Exception("Modes other than topk not yet available")

            # Convert logits to probabilities, with a fixed temperature
            texts_top = [self.commander.tokenizer.decode(idx) for idx in logits_top_idx]
            probs_top = torch.nn.functional.softmax(logits_top / self.t, dim=-1)

            # t_aux is used to sample the next token, probabilities are not stored
            probs_aux = torch.nn.functional.softmax(logits_top / t_aux, dim=-1)
            probs_aux = probs_aux.detach().numpy()
            next_idx = self._rng.choice(len(texts_top), p=probs_aux)

            d = {
                "texts": texts_top,
                "token_ids": logits_top_idx,
                "probs": probs_top,
                "selected_idx": next_idx,
                "selected_text": texts_top[next_idx],
            }
            self._data.loc[len(self._data)] = d

            output += texts_top[next_idx]
            output_ids = torch.cat(
                [output_ids, logits_top_idx[next_idx].reshape(1, -1)], dim=-1
            )

            if verbose:
                print(texts_top[next_idx], end="", flush=True)

        if verbose:
            print()
