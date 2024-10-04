import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class _ScoutViz:
    def __init__(self, output_scouting):
        self._output_scouting = output_scouting

    def prob_norm_hist(self, **kwargs):
        data = self._output_scouting.get_data()
        ax = data["prob_norm"].hist(**kwargs)
        return ax

    def prob_norm_kde(self, **kwargs):
        data = self._output_scouting.get_data()
        ax = sns.kdeplot(data=data, x="prob_norm", **kwargs)
        return ax

    def temp_prob_scatter(self, **kwargs):
        temp_setter = self._output_scouting.temp_setter

        prob_space = np.linspace(
            start=temp_setter.min_prob, stop=temp_setter.max_prob, num=100
        )
        temp_space = temp_setter.model_.predict(prob_space.reshape(-1, 1))

        if "ax" in kwargs:
            ax = kwargs.pop("ax")
        else:
            fig, ax = plt.subplots()

        # Plot regressor
        ax.plot(temp_space, prob_space)

        # Plot probabilities and actual t_aux
        ax.scatter(temp_setter._t_aux, temp_setter._probs, **kwargs)

        ax.set_xlabel("aux_T")
        ax.set_xlim(temp_setter.t_min, temp_setter.t_max)

        ax.set_ylabel("prob_norm")
        ax.set_ylim(0, temp_setter.max_prob)

        return ax
