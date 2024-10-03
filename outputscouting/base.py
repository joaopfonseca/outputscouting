import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import check_random_state
from ._commander import CentralCommand
from ._scout import Scout
from ._scheduler import TemperatureScheduler


class OutputScouting:
    def __init__(
        self,
        prompt,
        model,
        tokenizer,
        cooling_function="scheduler",
        t_min=0.01,
        t_max=2,
        power=2,
        n_scouts=500,
        k=10,
        p=None,
        max_length=np.inf,
        cuda=None,
        random_state=None,
        verbose=False,
    ):
        self.prompt = prompt
        self.model = model
        self.tokenizer = tokenizer
        self.cooling_function = cooling_function
        self.t_min = t_min
        self.t_max = t_max
        self.power = power
        self.n_scouts = n_scouts
        self.k = k
        self.p = p
        self.max_length = max_length
        self.cuda = cuda
        self.random_state = random_state
        self.verbose = verbose

    def standard_walk(self):
        if not hasattr(self, "_commander"):
            self._commander = CentralCommand(self.model, self.tokenizer)

        if not hasattr(self, "_rng"):
            self._rng = check_random_state(self.random_state)

        if self.cooling_function == "scheduler":
            self.cooling_schedule = TemperatureScheduler(
                start=self.start, end=self.end, power=self.power
            )
        elif self.cooling_function is not None:
            self.cooling_schedule = self.cooling_function(
                start=self.start, stop=self.end, power=self.power, num=self.n_scouts
            )
        else:
            self.cooling_schedule = [None] * self.n_scouts

        self.scouts = []
        for i in range(self.n_scouts):
            if self.cooling_function == "scheduler" and len(self.aux_temperatures) < 0:
                aux_T = self.cooling_schedule.get_temperature()
            elif self.cooling_function == "scheduler":
                aux_T = self.cooling_schedule.get_temperature(i)
            else:
                aux_T = self.cooling_schedule[i]
            scout = Scout(
                self.prompt,
                self._commander,
                k=self.k,
                aux_T=aux_T,
                max_length=self.max_length,
            )
            scout.standard_walk(verbose=self.verbose)
            prob = scout.get_data()['prob_norm']
            self.cooling_schedule.add_entry(prob, aux_T)
            self.scouts.append(scout)

        return self

    def get_data(self):
        return pd.DataFrame([scout.get_data() for scout in self.scouts])

    def plot_cooling_schedule(self, show=False):
        if self.cooling_function is not None:
            sns.lineplot(
                x=range(self.n_scouts),
                y=self.cooling_function(
                    start=self.start, stop=self.end, power=self.power, num=self.n_scouts
                ),
            )
        else:
            print("No cooling function provided.")

        if show:
            plt.show()
        return self

    def plot_prob_norm(self, include_duplicates=True, show=False):
        if include_duplicates:
            data = self.get_data()
        else:
            data = self.get_data().drop_duplicates(subset="phrase")

        ax = sns.kdeplot(data=data, x="prob_norm")
        if show:
            plt.show()
        else:
            return ax
