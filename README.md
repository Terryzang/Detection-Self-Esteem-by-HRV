# Detection-Self-Esteem-by-HRV
The resting state ECG and task state ECG files contain ECG segments recorded while participants were in resting and task states, respectively. These files are stored in .npy format and have a sampling rate of 2000Hz. The data is raw and unprocessed.

Feature_extraction.py includes code for filtering and preprocessing the ECG data, as well as for extracting HRV features. It processes all .npy files in the directory and saves the extracted HRV features in an Excel file.

rest_task_161.xlsx contains the HRV features extracted from the clean, preprocessed ECG data. This Excel file includes the dataset we used for machine learning, along with additional information such as gender, age, and self-esteem scale scores. Gender is one-hot encoded.

Model_SVM.py and Model_others.py contain the code for the four models used in our study. To prevent data leakage from the training set, we used pipelines. The results are saved in multiple Excel files, including the actual predictions from each independent experiment, as well as accuracy, precision, and recall scores for each experiment.

To run our code smoothly, you need to install the necessary libraries, including scikit-learn, numpy, pandas, and neurokit2.
