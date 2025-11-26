from dart.external.discount import harmonic_number
from dart.external.kl_divergence import compute_kl_divergence
from sklearn.preprocessing import KBinsDiscretizer
import warnings
import numpy as np
import dart.handler.other.textstat


class Calibration:
    """
    Class that calibrates recommender Calibration.
    Theory: https://dl.acm.org/doi/10.1145/3240323.3240372
    Implementation: http://ethen8181.github.io/machine-learning/recsys/calibration/calibrated_reco.html.
    """

    def __init__(self, config):
        n_bins = 5
        self.bins_discretizer = KBinsDiscretizer(encode='ordinal', n_bins=n_bins, strategy='uniform')
        warnings.filterwarnings("ignore", category=UserWarning)
        self.language = config['language']
        self.textstat = dart.handler.other.textstat.TextStatHandler(self.language)

    def compute_distr(self, items, adjusted=False, subcategory=False):
        """Compute the genre distribution for a given list of Items."""
        n = len(items)
        sum_one_over_ranks = harmonic_number(n)
        count = 0
        distr = {}
        for item in items:
            count += 1
            if subcategory:
                topic_freq = distr.get(item['subcategory'], 0.)
                distr[item['subcategory']] = topic_freq + 1 * 1 / count / sum_one_over_ranks if adjusted else topic_freq + 1 * 1 / n 
            else:
                topic_freq = distr.get(item['category'], 0.)
                distr[item['category']] = topic_freq + 1 * 1 / count / sum_one_over_ranks if adjusted else topic_freq + 1 * 1 / n

        return distr

    def compute_distr_complexity(self, arr, bins_discretizer, adjusted=False):
        """
            Args:
            Return"
        """
        n = len(arr)
        sum_one_over_ranks = harmonic_number(n)
        arr_binned = bins_discretizer.transform(arr)
        distr = {}
        if adjusted:
            for bin in list(range(bins_discretizer.n_bins)):
                for indx, ele in enumerate(arr_binned[:, 0]):
                    if ele == bin:
                        rank = indx + 1
                        bin_freq = distr.get(bin, 0.)
                        distr[bin] = bin_freq + 1 * 1 / rank / sum_one_over_ranks

        else:
            for bin in list(range(bins_discretizer.n_bins)):
                distr[bin] = round(np.count_nonzero(arr_binned == bin) / arr_binned.shape[0], 3)
        return distr

    def topic_divergence(self, reading_history, recommendation, subcategory=False):
        freq_rec = self.compute_distr(recommendation, adjusted=True, subcategory=subcategory)
        freq_history = self.compute_distr(reading_history, adjusted=True, subcategory=subcategory)
        divergence_with_discount = compute_kl_divergence(freq_history, freq_rec)

        freq_rec = self.compute_distr(recommendation, adjusted=False, subcategory=subcategory)
        freq_history = self.compute_distr(reading_history, adjusted=False, subcategory=subcategory)
        divergence_without_discount = compute_kl_divergence(freq_history, freq_rec)
        return [divergence_with_discount, divergence_without_discount]

    def _article_complexity(self, item):
        value = item.get('complexity')
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                value = None
        if value is None and self.textstat is not None:
            text = item.get('text') or ''
            if text:
                try:
                    return float(self.textstat.flesch_kincaid_score(text))
                except Exception:
                    return 0.0
        return 0.0

    def complexity_divergence(self, reading_history, recommendation):
        reading_history_complexity = np.array(
            [self._article_complexity(x) for x in reading_history],
            dtype=np.float64,
        ).reshape(-1, 1)
        recommendation_complexity = np.array(
            [self._article_complexity(x) for x in recommendation],
            dtype=np.float64,
        ).reshape(-1, 1)

        try:
            self.bins_discretizer.fit(reading_history_complexity)
        except ValueError:
            return [0.0, 0.0]

        distr_pool = self.compute_distr_complexity(reading_history_complexity, self.bins_discretizer, True)
        distr_recommendation = self.compute_distr_complexity(recommendation_complexity, self.bins_discretizer, True)
        divergence_with_discount = compute_kl_divergence(distr_pool, distr_recommendation)

        distr_pool = self.compute_distr_complexity(reading_history_complexity, self.bins_discretizer, False)
        distr_recommendation = self.compute_distr_complexity(recommendation_complexity, self.bins_discretizer, False)
        divergence_without_discount = compute_kl_divergence(distr_pool, distr_recommendation)
        return [divergence_with_discount, divergence_without_discount]

    def calculate(self, reading_history, recommendation, complexity = True, subcategory = False):
        if len(reading_history) > 0 and len(recommendation) > 0:
            topic_divergence = self.topic_divergence(reading_history, recommendation, subcategory)
            if complexity:
                complexity_divergence = self.complexity_divergence(reading_history, recommendation)
            else:
                complexity_divergence = 0
            return topic_divergence, complexity_divergence
        else:
            return
