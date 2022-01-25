import yaml
import argparse

from warnings import warn

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from joblib import dump


def multilabel_sample(y, size=1000, min_count=5, seed=None):
    """ Takes a matrix of binary labels `y` and returns
        the indices for a sample of size `size` if
        `size` > 1 or `size` * len(y) if size =< 1.
        The sample is guaranteed to have > `min_count` of
        each label.
    """
    try:
        if (np.unique(y).astype(int) != np.array([0, 1])).any():
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError('multilabel_sample only works with binary indicator matrices')

    if (y.sum(axis=0) < min_count).any():
        raise ValueError('Some classes do not have enough examples. Change min_count if necessary.')

    if size <= 1:
        size = np.floor(y.shape[0] * size)

    if y.shape[1] * min_count > size:
        msg = "Size less than number of columns * min_count, returning {} items instead of {}."
        warn(msg.format(y.shape[1] * min_count, size))
        size = y.shape[1] * min_count

    rng = np.random.RandomState(seed if seed is not None else np.random.randint(1))

    if isinstance(y, pd.DataFrame):
        choices = y.index
        y = y.values
    else:
        choices = np.arange(y.shape[0])

    sample_idxs = np.array([], dtype=choices.dtype)

    # first, guarantee > min_count of each label
    for j in range(y.shape[1]):
        label_choices = choices[y[:, j] == 1]
        label_idxs_sampled = rng.choice(label_choices, size=min_count, replace=False)
        sample_idxs = np.concatenate([label_idxs_sampled, sample_idxs])

    sample_idxs = np.unique(sample_idxs)

    # now that we have at least min_count of each, we can just random sample
    sample_count = int(size - sample_idxs.shape[0])

    # get sample_count indices from remaining choices
    remaining_choices = np.setdiff1d(choices, sample_idxs)
    remaining_sampled = rng.choice(remaining_choices,
                                   size=sample_count,
                                   replace=False)

    return np.concatenate([sample_idxs, remaining_sampled])


def multilabel_sample_dataframe(df, labels, size, min_count=5, seed=None):
    """ Takes a dataframe `df` and returns a sample of size `size` where all
        classes in the binary matrix `labels` are represented at
        least `min_count` times.
    """
    idxs = multilabel_sample(labels, size=size, min_count=min_count, seed=seed)
    return df.loc[idxs]


def multilabel_train_test_split(X, Y, size, min_count=5, seed=None):
    """ Takes a features matrix `X` and a label matrix `Y` and
        returns (X_train, X_test, Y_train, Y_test) where all
        classes in Y are represented at least `min_count` times.
    """
    index = Y.index if isinstance(Y, pd.DataFrame) else np.arange(Y.shape[0])

    test_set_idxs = multilabel_sample(Y, size=size, min_count=min_count, seed=seed)
    train_set_idxs = np.setdiff1d(index, test_set_idxs)

    test_set_mask = index.isin(test_set_idxs)
    train_set_mask = ~test_set_mask

    return (X[train_set_mask], X[test_set_mask], Y[train_set_mask], Y[test_set_mask])


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", 
        "--config", 
        dest="config",
        help="Path to Hyperparameters configuration file."
    )
    args = parser.parse_args()

    # Ensure a config was passed to the script.
    if not args.config:
        print("No Hyperparameters configuration file provided.")
        exit()

    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)
        df = pd.read_csv(config["data"]["train_path"], index_col=0, nrows=50_000)
        # Create the new DataFrame: numeric_data_only
        numeric_data_only = df[config["data"]["numeric_columns"]].fillna(-1000)
        # Get labels and convert to dummy variables: label_dummies
        label_dummies = pd.get_dummies(df[config["data"]["labels"]])
        # Create training and test sets
        X_train, X_test, y_train, y_test = multilabel_train_test_split(
            numeric_data_only,
            label_dummies,
            size=0.2, 
            seed=123,
            min_count=5
        )
        # Instantiate the classifier: clf
        clf = OneVsRestClassifier(LogisticRegression())
        # Fit the classifier to the training data
        clf.fit(X_train, y_train)
        # Print the accuracy
        print("Accuracy: {}".format(clf.score(X_test, y_test)))
        with open("metrics.txt", "w") as metrics_file:
            metrics_file.write("Accuracy: {}".format(clf.score(X_test, y_test)))
        # Persist the trained model
        dump(clf, 'one_vs_rest_classifier_v1.joblib')