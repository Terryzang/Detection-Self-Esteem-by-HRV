from imblearn.pipeline import make_pipeline as make_imb_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score , accuracy_score
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# loading data
data = pd.read_excel('E:\\rest_task_161.xlsx', sheet_name='rest+task')

features = data.iloc[:, 1:-1]
self_esteem = data.iloc[:, -1]
self_esteem_4 = pd.qcut(self_esteem, 2, labels=False)
low_self_esteem = (self_esteem_4 == 0).astype(int)
ids = data.iloc[:, 0]
print(features)


# define classifiers
PCA_number = None
classifiers = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', C=0.1, random_state=42),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'Naive Bayes': GaussianNB()
}

# create cross validation
outer_cv = StratifiedKFold(n_splits=10, shuffle=True)

# Traversal classifiers
for name, classifier in classifiers.items():
    print(f"\nProcessing {name}")

    # Initializes the result list
    overall_results = []

    # Run 60 independent experiments
    for experiment_num in range(60):
        print(f"\nExperiment {experiment_num + 1} / 60")
        results_list = []
        total_accuracy = []
        total_precision = []
        total_recall = []
        total_f1 = []
        fold_number = 1

        # using Pipeline with PCA
        pipeline = make_imb_pipeline(
            StandardScaler(),
            PCA(n_components=PCA_number),
            classifier
        )

        # Cross validation dataset partitioning
        for train_index, test_index in outer_cv.split(features, low_self_esteem):
            X_train, X_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = low_self_esteem[train_index], low_self_esteem[test_index]
            ids_test = ids.iloc[test_index]

            # The training the models
            pipeline.fit(X_train, y_train)
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)

            # Performance is calculated on the test set
            test_accuracy = accuracy_score(y_test, y_pred_test)
            test_precision = precision_score(y_test, y_pred_test, pos_label=1, zero_division=0)
            test_recall = recall_score(y_test, y_pred_test, pos_label=1, zero_division=0)
            test_f1 = f1_score(y_test, y_pred_test, pos_label=1, zero_division=0)


            # save results
            fold_results = pd.DataFrame({
                'Experiment': experiment_num + 1,
                'Fold': fold_number,
                'ID': ids_test,
                'True Label': y_test,
                'Predicted Label': y_pred_test,
                'ACC': (y_pred_test == y_test).astype(int)
            })
            results_list.append(fold_results)

            # ccumulate the result of each fold
            total_accuracy.append(test_accuracy)
            total_precision.append(test_precision)
            total_recall.append(test_recall)
            total_f1.append(test_f1)
            fold_number += 1


        # Concatenate all results
        all_results = pd.concat(results_list)
        all_results.to_excel(f'E:\\{name}_{PCA_number}_{experiment_num + 1}.xlsx', index=False)

        # Record overall results
        overall_results.append({
            'Experiment': experiment_num + 1,
            'Model': name,
            'Average Accuracy': np.mean(total_accuracy),
            'Average Precision': np.mean(total_precision),
            'Average Recall': np.mean(total_recall),
            'Average F1': np.mean(total_f1)
        })
    # Save the summary results of 60 experiments
    overall_df = pd.DataFrame(overall_results)
    overall_df.to_excel(f'E:\\{name}_PCA{PCA_number}.xlsx', index=False)
