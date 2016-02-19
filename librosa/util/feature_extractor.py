#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scikit-learn feature extraction integration"""

from .decorators import deprecated
from sklearn.base import BaseEstimator, TransformerMixin

__all__ = ['FeatureExtractor']


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Sci-kit learn wrapper class for feature extraction methods.

    This class acts as a bridge between feature extraction functions
    and scikit-learn pipelines.

    .. warning:: The `FeatureExtractor` object is deprecated as of 0.4.2, and will be
                 removed in 0.5.
                 Instead, use `sklearn.preprocessing.FunctionTransformer`.

    Attributes
    ----------
    function : function
        The feature extraction function to wrap.

        Example: `librosa.feature.melspectrogram`

    target : str or None
        If `None`, then `function` is called with the input
        data as the first positional argument.

        If `str`, then `function` is called with the input
        data as a keyword argument with key `target`.

    iterate : bool
        If `True`, then `function` is applied iteratively to each
        item of the input.

        If `False`, then `function` is applied to the entire data
        stream simultaneously.  This is useful for things like aggregation
        and stacking.

    kwargs : additional keyword arguments
        Parameters to be passed through to `function`

    Examples
    --------
    >>> import sklearn.pipeline
    >>> # Build a mel-spectrogram extractor
    >>> MS = librosa.util.FeatureExtractor(librosa.feature.melspectrogram,
    ...                                    sr=22050, n_fft=2048,
    ...                                    n_mels=128, fmax=8000)
    >>> # And a log-amplitude extractor
    >>> LA = librosa.util.FeatureExtractor(librosa.logamplitude,
    ...                                    ref_power=np.max)
    >>> # Chain them into a pipeline
    >>> Features = sklearn.pipeline.Pipeline([('MelSpectrogram', MS),
    ...                                       ('LogAmplitude', LA)])
    >>> # Load an audio file
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> # Apply the transformation to y
    >>> F = Features.transform([y])
    """

    @deprecated('0.4.2', '0.5')
    def __init__(self, function, target=None, iterate=True, **kwargs):
        '''FeatureExtractor constructor

        '''
        self.function = function
        self.target = target
        self.iterate = iterate
        self.kwargs = {}

        self.set_params(**kwargs)

    # Clobber _get_param_names here for transparency
    def _get_param_names(self):
        """Returns the parameters of the feature extractor as a dictionary."""
        temp_params = {'function': self.function, 'target': self.target}

        temp_params.update(self.kwargs)

        return temp_params

    # Wrap set_params to catch updates
    def set_params(self, **kwargs):
        """Update the parameters of the feature extractor."""

        # We don't want non-functional arguments polluting kwargs
        params = kwargs.copy()
        for k in ['function', 'target']:
            params.pop(k, None)

        self.kwargs.update(params)
        BaseEstimator.set_params(self, **kwargs)

    # We keep these arguments for compatibility, but don't use them.
    def fit(self, *args, **kwargs):  # pylint: disable=unused-argument
        """This function does nothing, and is provided for interface compatibility.

        .. note:: Since most `TransformerMixin` classes implement some
            statistical modeling (e.g., PCA), the `fit()` method is
            required.

            For the `FeatureExtraction` class, all parameters are fixed
            ahead of time, and no statistical estimation takes place.
        """

        return self

    # Variable name 'X' is for consistency with sklearn
    def transform(self, X):  # pylint: disable=invalid-name
        """Applies the feature transformation to an array of input data.

        Parameters
        ----------
        X : iterable
            Array or list of input data

        Returns
        -------
        X_transform : list
            In positional argument mode (`target=None`), then
            `X_transform[i] = function(X[i], [feature parameters])`

            If the `target` parameter was given, then
            `X_transform[i] = function(target=X[i], [feature parameters])`
        """

        if self.target is not None:
            # If we have a target, each element of X takes the keyword argument
            if self.iterate:
                return [self.function(**dict(list(self.kwargs.items())
                                             + list({self.target: i}.items())))
                        for i in X]
            else:
                return self.function(**dict(list(self.kwargs.items())
                                            + list({self.target: X}.items())))
        else:
            # Each element of X takes first position in function()
            if self.iterate:
                return [self.function(i, **self.kwargs) for i in X]
            else:
                return self.function(X, **self.kwargs)
