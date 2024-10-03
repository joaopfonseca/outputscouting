import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, uniform
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# https://stackoverflow.com/questions/53345583/python-numpy-exponentially-spaced-samples-between-a-start-and-end-value
def powerspace(start, stop, power, num):
    start = np.power(start, 1 / float(power))
    stop = np.power(stop, 1 / float(power))
    return np.power(np.linspace(start, stop, num=num), power)


def inverse_powerspace(start, stop, power, num):
    start = np.power(start, 1 / float(power))
    stop = np.power(stop, 1 / float(power))
    return np.power(start, power) - np.power(np.linspace(stop, start, num=num), power)


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
        min_prob=None,
        max_prob=None,
        degree=None,
        target_distribution=uniform(),
        mode="kde",
        bins=20,
    ):
        self.t_min = t_min
        self.t_max = t_max
        self.min_prob = min_prob
        self.max_prob = max_prob
        self.degree = degree
        self.bins = bins
        self.target_distribution = target_distribution
        self.mode = mode

        self._probs = np.array([])
        self._t_aux = np.array([])

        self._target_prob_history = []

    def add_point(self, prob, t_aux):
        # Add point
        self._probs = np.append(self._probs, prob).reshape(-1, 1)
        self._t_aux = np.append(self._t_aux, t_aux)

        # Update the stored max and min
        self.min_prob = min(self._t_aux)
        self.max_prob = max(self._t_aux)

        return True

    def fit_line(self, plot=True):

        if self.degree:
            model = Pipeline(
                [
                    ("poly", PolynomialFeatures(degree=self.degree)),
                    ("linear", LinearRegression(fit_intercept=True)),
                ]
            )
            model.fit(self._probs, self._t_aux)

        else:
            model = LinearRegression(fit_intercept=True).fit(self._probs, self._t_aux)

        prob_space = np.linspace(start=self.t_min, stop=self.t_max, num=100)
        temp_space = model.predict(prob_space.reshape(-1, 1))

        if plot:
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
                self.y, range=(self.min_prob, self.max_prob), bins=self.bins
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
            self.kde_ = gaussian_kde(self.y, bw_method="silverman")

            # test_var = np.linspace(0,1,100)
            # plt.plot(test_var, self.kde_(test_var))
            # plt.xlim(self.min_prob,self.max_prob)
            # plt.show()

            target_prob_norm = sample_from_pdf(
                lambda x: self.target_distribution.pdf(x) - self.kde_.pdf(x),
                n_samples=1,
                loc=[self.min_prob, self.max_prob],
            )[0]

        if self.degree is not None:
            idx = (np.abs(prob_space - target_prob_norm)).argmin()
            next_aux_T = temp_space[idx]
        else:
            next_aux_T = (target_prob_norm - model.intercept_) / model.coef_[0]

        if next_aux_T < self.t_min:
            next_aux_T = self.t_min
        elif next_aux_T > self.t_max:
            next_aux_T = self.t_max

        return next_aux_T, target_prob_norm