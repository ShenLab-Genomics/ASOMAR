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

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_recall_curve, auc

kfold = KFold(n_splits=10, shuffle=True)
acc_per_fold = []
loss_per_fold = []
auroc_fold = []

input_data = DataReader("C:/Users/yagao/Documents/ASO/data/train_select.csv")
(df, seqs, labels, efficacy) = input_data.load_train_set(encoding='one_hot', max_length=20)

features = ["concentration", "self_bind", "dG", "MFE", "ASOMFE", "TM","open_prob"]
category = ["modify"]
plain = ["open_pc", 'max_open_length']
categories = df["modify"]
auroc_fold = []
auc_pr_fold = []
for fold_no, (train_index, test_index) in enumerate(kfold.split(df)):

    trainX, testX = seqs[train_index], seqs[test_index]
    trainY, testY = labels[train_index], labels[test_index]

    aso_model = ASOdeep(feature_num=0, seq_shape=(trainX.shape[1], trainX.shape[2]))
    # ens_model = aso_model.create_combined_model()
    ens_model = aso_model.seq_only_model()

    ens_model.compile(loss=BinaryCrossentropy(),
                      optimizer='adam',
                      metrics=['accuracy'])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=0.001, patience=25, mode='min',
                                                  restore_best_weights=True)

    # history = ens_model.fit([trainAttrX, trainX], trainY,
    #                         validation_data=([testAttrX, testX], testY),
    #                         batch_size=64, epochs=500, verbose=0, callbacks=early_stop)
    history = ens_model.fit([trainX], trainY,
                            validation_data=([testX], testY),
                            batch_size=64, epochs=500, verbose=0, callbacks=early_stop)

    # scores = ens_model.evaluate([testAttrX, testX], testY, verbose=0)
    scores = ens_model.evaluate([testX], testY, verbose=0)
    print(f'Score for fold {fold_no + 1}: {ens_model.metrics_names[0]} of {scores[0]}; {ens_model.metrics_names[1]} of {scores[1] * 100}%')
    fold_no = fold_no+1
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    # preds = ens_model.predict([testAttrX, testX], verbose=0)
    preds = ens_model.predict([testX], verbose=0)
    probs = np.array(preds[:,1])
    y_test = np.array(testY[:,1])
    fpr, tpr, threshold = metrics.roc_curve(y_test, probs)
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)
    auc_pr_fold.append(pr_auc)
    auroc_fold.append(roc_auc)

    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})\n')
    print(f'> AUCROC: {np.mean(auroc_fold)} (+- {np.std(auroc_fold)})\n')
    print(f'> AUCPR: {np.mean(auc_pr_fold)} (+- {np.std(auc_pr_fold)})\n')