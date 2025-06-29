import numpy as np
from itertools import product

def leave_one_pair_out(PDL_features, ETL_features):
    assert PDL_features.shape == ETL_features.shape
    n_patients = PDL_features.shape[0]

    # Store combinations of indices (i, j): i from PDL, j from ETL
    combinations = list(product(range(n_patients), range(n_patients)))

    for i, j in combinations:
        # Create train sets by excluding the i-th PDL and j-th ETL patients
        X_PDL_train = np.delete(PDL_features, i, axis=0)
        X_ETL_train = np.delete(ETL_features, j, axis=0)
        
        # Create test set from the held-out pair
        X_test = np.stack([PDL_features[i], ETL_features[j]])  # shape: (2, 14, 600)
        y_test = np.array([0, 1])  # assuming 0=PDL, 1=ETL

        # Labels for training
        y_PDL_train = np.zeros(X_PDL_train.shape[0])
        y_ETL_train = np.ones(X_ETL_train.shape[0])

        # Combine training data
        X_train = np.concatenate([X_PDL_train, X_ETL_train], axis=0)
        y_train = np.concatenate([y_PDL_train, y_ETL_train], axis=0)

        yield X_train, y_train, X_test, y_test, (i, j)


def stratified_random_split(X, y, val_size=2, seed=None):
    np.random.seed(seed)
    X, y = np.array(X), np.array(y)

    # Get indices per class
    class0_idx = np.where(y == 0)[0]
    class1_idx = np.where(y == 1)[0]

    # Sample one from each class for validation
    val_class0 = np.random.choice(class0_idx, 1, replace=False)
    val_class1 = np.random.choice(class1_idx, 1, replace=False)
    val_idx = np.concatenate([val_class0, val_class1])

    # Remaining indices are for training
    train_idx = np.setdiff1d(np.arange(len(y)), val_idx)

    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]