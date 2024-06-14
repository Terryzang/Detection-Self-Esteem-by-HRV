from imblearn.pipeline import make_pipeline as make_imb_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score , accuracy_score
import pandas as pd
import numpy as np

# loading data
data = pd.read_excel('E:\\rest_task_161.xlsx', sheet_name='rest+task')

features = data.iloc[:, 1:-1]
self_esteem = data.iloc[:, -1]
self_esteem_4 = pd.qcut(self_esteem, 2, labels=False)
low_self_esteem = (self_esteem_4 == 0).astype(int)
ids = data.iloc[:, 0]
print(features)

# define kernels
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
PCA_number = 12

# define parameter grids
param_grids = {
    'linear': {'svc__C':[0.1,1,10], 'svc__kernel': ['linear']},
    'rbf': {'svc__C': [0.1,1,10],'svc__gamma': ['scale', 'auto'], 'svc__kernel': ['rbf']},
    'poly': {'svc__C':[0.1,1,10],'svc__gamma': ['scale', 'auto'],'svc__kernel': ['poly']},
    'sigmoid': {'svc__C':[0.1,1,10], 'svc__gamma': ['scale', 'auto'], 'svc__kernel': ['sigmoid']}
}

# define cross validation
inner_cv = StratifiedKFold(n_splits=10, shuffle=True)
outer_cv = StratifiedKFold(n_splits=10, shuffle=True)

# Traversal kernels
for kernel in kernels:
    print(f"\nProcessing {kernel} kernel")

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

        # Cross validation dataset partitioning
        for train_index, test_index in outer_cv.split(features, low_self_esteem):
            X_train, X_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = low_self_esteem[train_index], low_self_esteem[test_index]
            ids_test = ids.iloc[test_index]

            # using Pipeline with PCA and SVC
            pipeline = make_imb_pipeline(
                StandardScaler(),
                PCA(n_components=PCA_number),
                SVC(class_weight='balanced', probability=True)
            )

            # creat GridSearchCV
            grid = GridSearchCV(pipeline, param_grid=param_grids[kernel], cv=inner_cv, scoring='accuracy', verbose=1)
            # Perform a grid search and get the best parameters
            grid.fit(X_train, y_train)
            # Validate using the best model
            best_model = grid.best_estimator_

            # Performance is calculated on the test set
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
            recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

            # save results
            fold_results = pd.DataFrame({
                'Experiment': experiment_num + 1,
                'Fold': fold_number,
                'ID': ids_test,
                'True Label': y_test,
                'Predicted Label': y_pred,
                'best_estimator': str(grid.best_estimator_),
                'ACC': (y_pred == y_test).astype(int)
            })
            results_list.append(fold_results)

            # Accumulate the result of each fold
            total_accuracy.append(accuracy)
            total_precision.append(precision)
            total_recall.append(recall)
            total_f1.append(f1)
            fold_number += 1


        # Concatenate all results
        all_results = pd.concat(results_list)
        all_results.to_excel(f'E:\\SVM_{kernel}_{experiment_num + 1}.xlsx', index=False)

        # Record overall results
        overall_results.append({
            'Experiment': experiment_num + 1,
            'Kernel': kernel,
            'Average Accuracy': np.mean(total_accuracy),
            'Average Precision': np.mean(total_precision),
            'Average Recall': np.mean(total_recall),
            'Average F1': np.mean(total_f1)
        })
    # Save the summary results of 60 experiments
    overall_df = pd.DataFrame(overall_results)
    overall_df.to_excel(f'E:\\SVM_PCA{PCA_number}.xlsx', index=False)
