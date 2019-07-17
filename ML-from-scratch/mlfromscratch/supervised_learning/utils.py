def divide_on_feature(X, feature_i, threshold):
    """Split data on feature i by threshold
    :X: Dataset
    :feature_i: feature i
    :threshold: value
    :returns: X1, X2
    """
    idx1 = X[:, feature_i] <= threshold
    idx2 = X[:, feature_i] > threshold
    return X[idx1], X[idx2]
