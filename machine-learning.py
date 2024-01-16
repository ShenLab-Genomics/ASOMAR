from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from data import *
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from sklearn.metrics import matthews_corrcoef, fbeta_score, accuracy_score, recall_score, confusion_matrix
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import KFold
import pandas as pd

data = pd.read_csv("C:/Users/yagao/Documents/ASO/data/train.csv")
labels = data.loc[:, 'Classify']
sequences = data.loc[:, 'Sequence'].tolist()
dataset = {}

for i, seq in enumerate(sequences):
    # split into nucleotides, remove tab characters
    nucleotides = list(seq)
    nucleotides = [x for x in nucleotides if x != '\t']
    while (len(nucleotides) < 20):
        nucleotides.append(' ')
    nucleotides.append(labels[i])
    # add to dataset
    dataset[i] = nucleotides

df = pd.DataFrame(dataset).T
df.rename(columns={20: 'Class'}, inplace=True)
series = []

for name in df.columns:
    series.append(df[name].value_counts())
info = pd.DataFrame(series)
details = info.T

numerical_df = pd.get_dummies(df)
numerical_df.head()

df = numerical_df.drop(columns=['Class_1'])

df.rename(columns={'Class_0': 'Class'}, inplace=True)

X = df.drop(['Class'], axis=1).to_numpy()
y = df['Class'].to_numpy()

input_data = DataReader("C:/Users/yagao/Documents/ASO/data/train.csv")
(df, seqs, labels, efficacy) = input_data.load_train_set(encoding='one_hot', max_length=20)
trainAttrX, tempAttrX, trainX, tempX, train_label, temp_label = train_test_split(df, X, y,
                                                                                 test_size=0.2,
                                                                                 shuffle=True,
                                                                                 random_state=42)
valAttrX, testAttrX, valX, testX, val_label, test_label = train_test_split(tempAttrX, tempX,
                                                                           temp_label,
                                                                           test_size=0.5,
                                                                           shuffle=True,
                                                                           random_state=42)
features = ["concentration", "self_bind", "open_prob", "dG", "MFE", "ASOMFE", "TM"]
category = ["modify"]
plain = ["max_open_length", "open_pc"]
categories = df["modify"]
feature_processor = NoneSeqFeatureProcessor(continuous=features, category=category, plain=plain)
_, testAttrX = feature_processor.process_train_features(trainAttrX, testAttrX, categories=categories)
trainAttrX, valAttrX = feature_processor.process_train_features(trainAttrX, valAttrX, categories=categories)
X_train = np.hstack([trainX, trainAttrX])
y_train = train_label
X_test = np.hstack([testX, testAttrX])
y_test = test_label

# Split the data into training and test dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
# scoring = 'accuracy'

# Define models to train
# names = ['K Nearest Neighbors', 'Gaussian Process', 'Decision Tree', 'Random Forest',
#          'Neural Network', 'AdaBoost', 'Naive Bayes', 'SVM Linear', 'SVM RBF', 'SVM Sigmoid']

names = ['KNN', 'Decision Tree', 'Random Forest',
         'AdaBoost', 'Naive Bayes', 'SVM']
classifiers = [
    KNeighborsClassifier(n_neighbors=10),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10, n_estimators=10, max_features=5),
    # MLPClassifier(alpha=1, max_iter=500),
    AdaBoostClassifier(),
    GaussianNB(),
    SVC(kernel='linear', probability=True),
    # SVC(kernel='rbf', probability=True),
    # SVC(kernel='sigmoid', probability=True)
]

models = zip(names, classifiers)
results = []
names = []
kfold = KFold(n_splits=10, shuffle=True)
roc_values = []
pr_values = []
for name, model in models:
    history = model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, predictions)
    precision, recall, _ = precision_recall_curve(y_test, predictions)
    # Append FPR, TPR values with model name to the list
    for i in range(len(fpr)):
        roc_values.append([name, fpr[i], tpr[i]])

    # Append precision, recall values with model name to the list
    for i in range(len(precision)):
        pr_values.append([name, precision[i], recall[i]])

# Convert lists to DataFrames
roc_df = pd.DataFrame(roc_values, columns=['Model', 'FPR', 'TPR'])
pr_df = pd.DataFrame(pr_values, columns=['Model', 'Precision', 'Recall'])

# Write DataFrames to CSV files
roc_df.to_csv('ML_roc_values.csv', index=False)
pr_df.to_csv('ML_pr_values.csv', index=False)

# for name, model in models:
#     # auc_roc_fold = []
#     # auc_pr_fold = []
#     gmean_fold = []
#     # mcc_fold = []
#     # fbeta_fold = []
#     # accuracy_fold = []
#     # sensitivity_fold = []
#     # specificity_fold = []
#     for fold_no, (train_index, test_index) in enumerate(kfold.split(X)):
#         history = model.fit(X[train_index], y[train_index])
#         predictions = model.predict(X[test_index])
#         fpr, tpr, _ = metrics.roc_curve(y[test_index], predictions)
#         precision, recall, _ = precision_recall_curve(y[test_index], predictions)
#         roc_auc = metrics.auc(fpr, tpr)
#         pr_auc = auc(recall, precision)
#         gmean = geometric_mean_score(y[test_index], predictions)
# mcc = matthews_corrcoef(y[test_index], predictions)
# fbeta = fbeta_score(y[test_index], predictions, beta=1)
# accuracy = accuracy_score(y[test_index], predictions)
# sensitivity = recall_score(y[test_index], predictions)
# tn, fp, fn, tp = confusion_matrix(y[test_index], predictions).ravel()
# specificity = tn / (tn+fp)
# auc_roc_fold.append(roc_auc)
# auc_pr_fold.append(pr_auc)
# gmean_fold.append(gmean)
# mcc_fold.append(mcc)
# fbeta_fold.append(fbeta)
# accuracy_fold.append(accuracy)
# sensitivity_fold.append(sensitivity)
# specificity_fold.append(specificity)
#     results.append({
#         'name': name,
#         'auc_roc_mean': np.mean(auc_roc_fold),
#         'auc_roc_std': np.std(auc_roc_fold),
#         'auc_pr_mean': np.mean(auc_pr_fold),
#         'auc_pr_std': np.std(auc_pr_fold),
#         'gmean_mean': np.mean(gmean_fold),
#         'gmean_std': np.std(gmean_fold),
#         'mcc_mean': np.mean(mcc_fold),
#         'mcc_std': np.std(mcc_fold),
#         'fbeta_mean': np.mean(fbeta_fold),
#         'fbeta_std': np.std(fbeta_fold),
#         'accuracy_mean': np.mean(accuracy_fold),
#         'accuracy_std': np.std(accuracy_fold),
#         'sensitivity_mean': np.mean(sensitivity_fold),
#         'sensitivity_std': np.std(sensitivity_fold),
#         'specificity_mean': np.mean(specificity_fold),
#         'specificity_std': np.std(specificity_fold)
#     })
#
# df = pd.DataFrame(results)
# df_long = pd.melt(df, id_vars=['name'], var_name='metric', value_name='score')
# df_long.to_csv('C:/Users/yagao/Documents/ASO/data/model_metrics.csv', index=False)

# models = zip(names, classifiers)
# # Test the algorithm on the validation dataset
# preds = []
# for name, model in models:
#     model.fit(X_train, y_train)
#     predictions = model.predict_proba(X_test)
#     preds.append(predictions)
#     #
#     # print(name)
#     # print(accuracy_score(y_test, predictions))
#     # print(classification_report(y_test, predictions))
#
#
# len(preds[2])
# a = np.array(preds[2])
# len(preds)
# np.save('machine_learning.npy', preds, allow_pickle=True)
#
# preds = np.load('machine_learning.npy')
#
# np.savez.('inputs.py',  X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
#
#
# for pred, name in zip(preds, names):
#     fpr, tpr, _ = metrics.roc_curve(y_test, pred[::,1])
#     precision, recall, _ = precision_recall_curve(y_test, pred[::,1])
#     roc_auc = metrics.auc(fpr, tpr)
#     pr_auc = auc(recall, precision)
#     plt.plot(fpr, tpr, label=name + ", AUC=" + str(roc_auc))
#     msg = '{0}:  {1}  ({2})'.format(name,roc_auc)
#     print(msg)
# plt.legend()
#
# for pred, name in zip(preds, names):
#     precision, recall, _ = precision_recall_curve(y_test, pred[::,1])
#     pr_auc = auc(recall, precision)
#     plt.plot(recall, precision, 'b', label=f'PR Curve (AUC = {pr_auc:.2f})')
# plt.legend()
