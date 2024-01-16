from sklearn.model_selection import KFold
from data import *
from models import ASOdeep
import tensorflow as tf
from keras.models import Model
from keras.layers import concatenate
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Input, BatchNormalization, GRU, Bidirectional
import sklearn.metrics as metrics
import numpy as np
import numpy as np
from sklearn.metrics import matthews_corrcoef, fbeta_score, accuracy_score, recall_score, confusion_matrix
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_recall_curve, auc

kfold = KFold(n_splits=10, shuffle=True)

auroc_fold = []

input_data = DataReader("C:/Users/yagao/Documents/ASO/data/train.csv")
(df, seqs, labels, efficacy) = input_data.load_train_set(encoding='one_hot', max_length=20)
# res = open("C:/Users/yagao/Documents/ASO/data/cnn_crossfold_new.txt", "a")

features = ["concentration", "self_bind", "dG", "MFE", "ASOMFE", "TM","open_prob"]
category = ["modify"]
plain = ["open_pc", 'max_open_length']
categories = df["modify"]
# for f in features:

auc_roc_fold = []
auc_pr_fold = []
gmean_fold = []
mcc_fold = []
fbeta_fold = []
accuracy_fold = []
sensitivity_fold = []
specificity_fold = []
results = []

for fold_no, (train_index, test_index) in enumerate(kfold.split(df)):

    feature_processor = NoneSeqFeatureProcessor(continuous=features, category=category, plain=plain)
    trainAttrX, testAttrX = feature_processor.process_train_features(df.iloc[train_index], df.iloc[test_index], categories=categories)
    trainX, testX = seqs[train_index], seqs[test_index]
    trainY, testY = labels[train_index], labels[test_index]

    aso_model = ASOdeep(feature_num=trainAttrX.shape[1], seq_shape=(trainX.shape[1], trainX.shape[2]))
    ens_model = aso_model.create_combined_model()

    ens_model.compile(loss=BinaryCrossentropy(),
                      optimizer='adam',
                      metrics=['accuracy'])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=0.001, patience=25, mode='min',
                                                  restore_best_weights=True)

    history = ens_model.fit([trainAttrX, trainX], trainY,
                            validation_data=([testAttrX, testX], testY),
                            batch_size=64, epochs=500, verbose=0, callbacks=early_stop)

    scores = ens_model.evaluate([testAttrX, testX], testY, verbose=0)
    print(f'Score for fold {fold_no + 1}: {ens_model.metrics_names[0]} of {scores[0]}; {ens_model.metrics_names[1]} of {scores[1] * 100}%')
    fold_no = fold_no+1
    preds = ens_model.predict([testAttrX, testX], verbose=0)
    probs = np.array(preds[:,1])
    threshold = 0.5
    binary_preds = [1 if prob > threshold else 0 for prob in probs]
    y_test = np.array(testY[:,1])
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
    precision, recall, _ = precision_recall_curve(y_test, probs)
    roc_auc = metrics.auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    gmean = geometric_mean_score(y_test, binary_preds)
    mcc = matthews_corrcoef(y_test, binary_preds)
    fbeta = fbeta_score(y_test, binary_preds, beta=1)
    accuracy = accuracy_score(y_test, binary_preds)
    sensitivity = recall_score(y_test, binary_preds)
    tn, fp, fn, tp = confusion_matrix(y_test, binary_preds).ravel()
    specificity = tn / (tn + fp)
    auc_roc_fold.append(roc_auc)
    auc_pr_fold.append(pr_auc)
    gmean_fold.append(gmean)
    mcc_fold.append(mcc)
    fbeta_fold.append(fbeta)
    accuracy_fold.append(accuracy)
    sensitivity_fold.append(sensitivity)
    specificity_fold.append(specificity)
    break

# results.append({
#     'auc_roc': {'mean': np.mean(auc_roc_fold), 'std': np.std(auc_roc_fold)},
#     'auc_pr': {'mean': np.mean(auc_pr_fold), 'std': np.std(auc_pr_fold)},
#     'gmean': {'mean': np.mean(gmean_fold), 'std': np.std(gmean_fold)},
#     'mcc': {'mean': np.mean(mcc_fold), 'std': np.std(mcc_fold)},
#     'fbeta': {'mean': np.mean(fbeta_fold), 'std': np.std(fbeta_fold)},
#     'accuracy': {'mean': np.mean(accuracy_fold), 'std': np.std(accuracy_fold)},
#     'sensitivity': {'mean': np.mean(sensitivity_fold), 'std': np.std(sensitivity_fold)},
#     'specificity': {'mean': np.mean(specificity_fold), 'std': np.std(specificity_fold)}
# })
# df = pd.DataFrame(results)
# df.to_csv('C:/Users/yagao/Documents/ASO/data/cnn_metrics.csv', index=False)
# print('Average scores for all folds\n')
#
# res.write(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})\n')
# res.write(f'> AUCROC: {np.mean(auroc_fold)} (+- {np.std(auroc_fold)})\n')
# res.write(f'> AUCPR: {np.mean(auc_pr_fold)} (+- {np.std(auc_pr_fold)})\n')
