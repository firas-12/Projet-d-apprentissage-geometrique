

def evaluate_embedding_node_classification_svm(embedding_matrix, labels, train_ratio=0.1, norm=True, seed=0, n_repeats=10):
    """Evaluate the node embeddings on the node classification task using SVM."""
    
    if norm:
        embedding_matrix = normalize(embedding_matrix)

    # Initialize SVM classifier
    svm_clf = SVC(kernel='linear', random_state=seed)

    results = []
    for it_seed in range(n_repeats):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=seed + it_seed)
        split_train, split_test = next(sss.split(embedding_matrix, labels))

        features_train = embedding_matrix[split_train]
        features_test = embedding_matrix[split_test]
        labels_train = labels[split_train]
        labels_test = labels[split_test]

        # Fit the SVM model
        svm_clf.fit(features_train, labels_train)

        # Make predictions
        y_pred = svm_clf.predict(features_test)

        # Calculate f1-scores
        f1_micro = f1_score(labels_test, y_pred, average='micro')
        f1_macro = f1_score(labels_test, y_pred, average='macro')

        results.append([f1_micro, f1_macro])

        # Print classification report
        print(f"Classification Report for SVM (seed {it_seed + 1}):")
        print(classification_report(labels_test, y_pred))

    # Convert results to numpy array and calculate mean and standard deviation
    results = np.array(results)
    mean_f1_scores = results.mean(0)
    std_f1_scores = results.std(0)

    # Print average f1-scores
    print(f"\nAverage F1 Scores for SVM:")
    print(f"Micro F1: {mean_f1_scores[0]:.4f} ± {std_f1_scores[0]:.4f}")
    print(f"Macro F1: {mean_f1_scores[1]:.4f} ± {std_f1_scores[1]:.4f}")

    return mean_f1_scores, std_f1_scores
