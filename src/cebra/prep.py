import numpy as np


def split_cebra_dataset(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.18,
    shuffle: bool = True,
    random_state: int = 42,
):
    """Split the data into training and testing sets based on unique classes in trials."""
    from sklearn.model_selection import train_test_split

    time_in_trial = y[:, 0]
    class_label = y[:, 1].astype(int)
    ntimes = y.shape[0]

    trial_id = np.zeros(ntimes, dtype=int)
    current_trial = 0
    trial_id[0] = 0

    for i in range(1, ntimes):
        if time_in_trial[i] < time_in_trial[i - 1]:
            current_trial += 1
        trial_id[i] = current_trial
    ntrials = current_trial + 1
    unique_trials = np.arange(ntrials)

    trial_classes = []
    for tr in unique_trials:
        idx = trial_id == tr
        classes = class_label[idx]
        vals, counts = np.unique(classes, return_counts=True)
        majority_class = vals[np.argmax(counts)]
        trial_classes.append(majority_class)
    trial_classes = np.array(trial_classes)

    trian_trials, test_trials = train_test_split(
        unique_trials,
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
        stratify=trial_classes,
    )
    train_mask = np.isin(trial_id, trian_trials)
    test_mask = np.isin(trial_id, test_trials)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    return X_train, X_test, y_train, y_test
