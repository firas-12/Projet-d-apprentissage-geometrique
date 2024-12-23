
from sklearn.preprocessing import normalize # type: ignore
from sklearn.model_selection import StratifiedShuffleSplit # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
import numpy as np
from sklearn.metrics import classification_report,f1_score # type: ignore

def evaluate_embedding_node_classification(embedding_matrix, labels, train_ratio=0.1, norm=True, seed=0, n_repeats=10):
    """Evaluate the node embeddings on the node classification task..

    :param embedding_matrix: np.ndarray, shape [n_nodes, embedding_dim]
        Embedding matrix
    :param labels: np.ndarray, shape [n_nodes]
        The ground truth labels
    :param train_ratio: float
        The fraction of labels to use for training
    :param norm: bool
        Whether to normalize the embeddings
    :param seed: int
        Random seed
    :param n_repeats: int
        Number of times to repeat the experiment
    :return: [float, float], [float, float]
        The mean and standard deviation of the f1_scores
    """
    if norm:
        embedding_matrix = normalize(embedding_matrix)

    results = []
    for it_seed in range(n_repeats):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=seed + it_seed)
        split_train, split_test = next(sss.split(embedding_matrix, labels))

        features_train = embedding_matrix[split_train]
        features_test = embedding_matrix[split_test]
        labels_train = labels[split_train]
        labels_test = labels[split_test]
        lr = LogisticRegression(solver='lbfgs', max_iter=1000, multi_class='auto')
        lr.fit(features_train, labels_train)

        lr_z_predict = lr.predict(features_test)

        # Calculate F1 scores
        f1_micro = f1_score(labels_test, lr_z_predict, average='micro')
        f1_macro = f1_score(labels_test, lr_z_predict, average='macro')

        results.append([f1_micro, f1_macro])

        # Print the classification report for the current fold
        print(f"Classification Report for seed {it_seed + 1}:")
        print(classification_report(labels_test, lr_z_predict))

    results = np.array(results)

    # Calculate and return the mean and std of f1 scores
    mean_f1_scores = results.mean(0)
    std_f1_scores = results.std(0)

    print("\nAverage F1 Scores across all repeats:")
    print(f"Micro F1: {mean_f1_scores[0]:.4f} ± {std_f1_scores[0]:.4f}")
    print(f"Macro F1: {mean_f1_scores[1]:.4f} ± {std_f1_scores[1]:.4f}")

    return mean_f1_scores, std_f1_scores
