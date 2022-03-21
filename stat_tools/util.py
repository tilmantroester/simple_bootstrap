import numpy as np

import scipy.interpolate
import scipy.optimize


def create_interpolated_cdf(samples, weights,
                            threshold=None, reverse=False,
                            start_at_zero=True):
    unique_samples, unique_inverse_idx = np.unique(samples,
                                                   return_inverse=True)
    if len(unique_samples) != len(samples):
        unique_weights = np.zeros_like(unique_samples)
        np.add.at(unique_weights, unique_inverse_idx, weights)
        samples = unique_samples
        weights = unique_weights

    if threshold is None:
        selection = np.ones_like(samples, dtype=bool)
    else:
        selection = samples <= threshold if reverse else samples >= threshold
    if reverse:
        sample_sort_idx = np.argsort(samples[selection])
        cdf = np.cumsum(weights[selection][sample_sort_idx])
        cdf = cdf[-1] - cdf
        if not start_at_zero:
            cdf += weights[selection][sample_sort_idx[-1]]
    else:
        sample_sort_idx = np.argsort(samples[selection])
        cdf = np.cumsum(weights[selection][sample_sort_idx])
        if start_at_zero:
            cdf -= cdf[0]

    if len(cdf) < 2:
        # Only one sample
        return lambda x: np.zeros_like(x)

    cdf_func = scipy.interpolate.InterpolatedUnivariateSpline(
                    x=samples[selection][sample_sort_idx],
                    y=cdf, k=1, ext=3)
    return cdf_func


def weighted_median(a, weights):
    if len(a) == 1:
        return a
    if np.all(np.isclose(a, a[0])):
        return a[0]

    if weights.sum()/weights.max()-1 < 1e-2:
        return a[np.argmax(weights)]

    l_cum = create_interpolated_cdf(a, weights, start_at_zero=False)
    r_cum = create_interpolated_cdf(a, weights, reverse=True,
                                    start_at_zero=False)

    return scipy.optimize.root_scalar(lambda x: l_cum(x) - r_cum(x),
                                      bracket=(a.min(), a.max())).root
