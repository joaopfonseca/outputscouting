import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, uniform
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def sample_from_pdf(pdf, n_samples=1, loc=[0, 1]):
    candidates = (uniform().rvs(size=1000 * n_samples) * (loc[1] - loc[0])) + loc[0]
    probs = pdf(candidates)
    probs = np.clip(probs, 0, 1)
    return np.random.choice(candidates, size=n_samples, p=(probs / probs.sum()))


class AuxTemperatureSetter:
    def __init__(
        self,
        t_min=0.01,
        t_max=2,
        degree=None,
        target_distribution=uniform(),
        mode="kde",
        bins=20,
    ):
        self.t_min = t_min
        self.t_max = t_max
        self.degree = degree
        self.bins = bins
        self.target_distribution = target_distribution
        self.mode = mode

        self._target_prob_history = []

    def add_point(self, prob, t_aux):
        # Add point
        self._probs = np.append(self._probs, prob).reshape(-1, 1)
        self._t_aux = np.append(self._t_aux, t_aux)

        # Update the stored max and min
        self.min_prob = min(self._probs.flatten())
        self.max_prob = max(self._probs.flatten())

    def get_temperature(self, plot=False):

        if not hasattr(self, "_probs"):
            self._probs = np.array([])
            self._t_aux = np.array([])

        if len(self._probs) == 0:
            return self.t_min
        elif len(self._probs) == 1:
            return self.t_max

        if self.degree:
            self.model_ = Pipeline(
                [
                    ("poly", PolynomialFeatures(degree=self.degree)),
                    ("linear", LinearRegression(fit_intercept=True)),
                ]
            )
            self.model_.fit(self._probs, self._t_aux)

        else:
            self.model_ = LinearRegression(fit_intercept=True).fit(
                self._probs, self._t_aux
            )

        if plot:
            prob_space = np.linspace(start=self.t_min, stop=self.t_max, num=100)
            temp_space = self.model_.predict(prob_space.reshape(-1, 1))
            # Plot regressor
            plt.plot(prob_space, temp_space)

            # Plot probabilities and actual t_aux
            plt.scatter(self._probs, self._t_aux)

            plt.xlabel("aux_T")
            plt.xlim(self.t_min, self.t_max)

            plt.ylabel("prob_norm")
            plt.ylim(0, self.max_prob)

            plt.show()

        if self.mode == "bins":
            hist, edges = np.histogram(
                self._probs, range=(self.min_prob, self.max_prob), bins=self.bins
            )

            if len(self._target_prob_history) > 0:
                hist_t, edges_t = np.histogram(
                    self._target_prob_history,
                    range=(self.min_prob, self.max_prob),
                    bins=self.bins,
                )
                hist = hist + hist_t

            argmin = np.random.choice(np.where(hist == hist.min())[0])

            target_prob_norm = np.random.uniform(edges[argmin], edges[argmin + 1], 1)[0]
            self._target_prob_history.append(target_prob_norm)

        elif self.mode == "kde":
            self.kde_ = gaussian_kde(self._probs.T, bw_method="silverman")

            # test_var = np.linspace(0,1,100)
            # plt.plot(test_var, self.kde_(test_var))
            # plt.xlim(self.min_prob,self.max_prob)
            # plt.show()

            target_prob_norm = sample_from_pdf(
                lambda x: self.target_distribution.pdf(x) - self.kde_.pdf(x),
                n_samples=1,
                loc=[self.min_prob, self.max_prob],
            )[0]

        t_aux = self.model_.predict(np.array([[target_prob_norm]]))[0]

        if t_aux < self.t_min:
            t_aux = self.t_min
        elif t_aux > self.t_max:
            t_aux = self.t_max

        return t_aux
