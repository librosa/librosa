#!/usr/bin/env python
""" Decomposition """

import sklearn.decomposition


def decompose(X, n_components=None, NMF=None):
    """Decompose the feature matrix with non-negative matrix factorization

    :parameters:
        - X : np.ndarray
            feature matrix (d-by-t)
        - n_components : int > 0 or None
            number of components, if None then all d components are used
        - NMF : any instance which implements fit_transform()
            If None, use sklearn.decomposition.NMF by default
            Otherwise, NMF.fit_transform() should take X as input, and returns
            transformed X_new, where X ~= NMF.components_.dot(X_new)

    :returns:
        - components: np.ndarray
            dictionary matrix (d-by-n_components)
        - X_new: np.ndarray
            transformed matrix/activation matrix (n_components-by-t)

    """

    if NMF is None:
        NMF = sklearn.decomposition.NMF(n_components=n_components)
        H = NMF.fit_transform(X.T)
        return (NMF.components_.T, H.T)

    H = NMF.fit_transform(X)
    return (NMF.components_, H)
