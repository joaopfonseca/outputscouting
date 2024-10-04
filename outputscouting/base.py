import numpy as np
import pandas as pd
from scipy.stats import uniform
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import check_random_state
from ._command import CentralCommand
from ._scout import Scout
from ._temp_setter import AuxTemperatureSetter
from ._visualization import _ScoutViz


class OutputScouting:
    def __init__(
        self,
        prompt,
        model,
        tokenizer,
        t=0.5,
        t_min=0.01,
        t_max=2,
        degree=3,
        target_distribution=uniform(),
        mode="kde",
        bins=20,
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
        self.t = t
        self.t_min = t_min
        self.t_max = t_max
        self.degree = degree
        self.target_distribution = target_distribution
        self.mode = mode
        self.bins = bins
        self.k = k
        self.p = p
        self.max_length = max_length
        self.cuda = cuda
        self.random_state = random_state
        self.verbose = verbose

        self.plot = _ScoutViz(self)

    def explore(self, n_scouts=1):
        if not hasattr(self, "_commander"):
            self._commander = CentralCommand(
                self.model, self.tokenizer, k=self.k, p=self.p, cuda=self.cuda
            )

        if not hasattr(self, "_rng"):
            self._rng = check_random_state(self.random_state)

        if not hasattr(self, "temp_setter"):
            self.temp_setter = AuxTemperatureSetter(
                t_min=self.t_min,
                t_max=self.t_max,
                degree=self.degree,
                target_distribution=self.target_distribution,
                mode=self.mode,
                bins=self.bins,
            )

        if not hasattr(self, "scouts"):
            self.scouts = []

        for i in range(n_scouts):
            t_aux = self.temp_setter.get_temperature()
            scout = Scout(
                self.prompt,
                self._commander,
                t=self.t,
                t_aux=t_aux,
                max_length=self.max_length,
            )
            scout.explore(verbose=self.verbose)
            prob = scout.get_data()["prob_norm"]
            self.temp_setter.add_point(prob, t_aux)
            self.scouts.append(scout)

        return self

    def get_data(self):
        return pd.DataFrame([scout.get_data() for scout in self.scouts])
