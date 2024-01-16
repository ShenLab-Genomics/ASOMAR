import pandas as pd
from keras import metrics
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.losses import BinaryCrossentropy
from models import ASOdeep
from data import DataReader, NoneSeqFeatureProcessor
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, r2_score


# compute AUC-ROC
def plot_AUC_ROC(preds, testY, plot=True):
    probs = np.array(preds[:, 1])
    y_test = np.array(testY[:, 1])
    fpr, tpr, threshold = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    print(f'AUCROC = {roc_auc:.2f}')
    if plot:
        plt.figure(figsize=(8, 6))
        plt.rcParams.update({'font.size': 22})
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


# compute AUC-PR
def plot_AUC_PR(preds, testY, plot=True):
    probs = np.array(preds[:, 1])
    y_test = np.array(testY[:, 1])
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)
    print(f'AUCPR = {pr_auc:.2f}')
    # Create the plot
    if plot:
        plt.figure(figsize=(8, 6))
        plt.rcParams.update({'font.size': 22})
        plt.plot(recall, precision, 'b', label=f'PR Curve (AUC = {pr_auc:.4f})')
        plt.title('Precision-Recall Curve')
        plt.plot([0, 1], [0.5, 0.5], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='lower left')
        plt.show()


def plot_predictions(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title(f'R-squared: {r2:.2f}')
    plt.show()


def plot_history(history):
    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.tight_layout()
    plt.show()


def main():
    input_data = DataReader("C:/Users/yagao/Documents/ASO/data/train.csv")
    (df, seqs, labels, efficacy) = input_data.load_train_set(encoding='one_hot', max_length=20)

    # gene_list = df['TargetGene'].value_counts().index
    # for g in gene_list:
    # test_index = df[df['TargetGene'] == g].index
    # train_index = df[df['TargetGene'] != g].index

    # '''
    trainAttrX, tempAttrX, trainX, tempX, train_label, temp_label, train_eff, temp_eff = train_test_split(df, seqs,
                                                                                                          labels,
                                                                                                          efficacy,
                                                                                                          test_size=0.2,
                                                                                                          shuffle=True,
                                                                                                          random_state=42)
    valAttrX, testAttrX, valX, testX, val_label, test_label, val_eff, test_eff = train_test_split(tempAttrX, tempX,
                                                                                                  temp_label,
                                                                                                  temp_eff,
                                                                                                  test_size=0.5,
                                                                                                  shuffle=True,
                                                                                                  random_state=42)
    # '''
    # train_set = df[df['TargetGene'] != g]
    # trainX = seqs[train_index]
    # train_label = labels[train_index]
    # train_eff = efficacy[train_index]
    # test_set = df[df['TargetGene'] == g]
    # testX = seqs[test_index]
    # test_label = labels[test_index]
    # test_eff = efficacy[test_index]

    # train_set, val_set, trainX, valX, train_label, val_label, train_eff, val_eff = train_test_split(train_set, trainX,
    #                                                                                                   train_label,
    #                                                                                                   train_eff,
    #                                                                                                   test_size=0.2,
    #                                                                                                   shuffle=True,
    #                                                                                                   random_state=42)

    features = ["concentration", "self_bind", "open_prob", "dG", "MFE", "ASOMFE", "TM"]
    category = ["modify"]
    plain = ["max_open_length", "open_pc"]
    categories = df["modify"]

    feature_processor = NoneSeqFeatureProcessor(continuous=features, category=category, plain=plain)
    _, testAttrX = feature_processor.process_train_features(trainAttrX, testAttrX, categories=categories)
    trainAttrX, valAttrX = feature_processor.process_train_features(trainAttrX, valAttrX, categories=categories)

    # trainAttrX, testAttrX = feature_processor.process_train_features(train_set, test_set, categories=categories)
    # temp_val, valAttrX = feature_processor.process_train_features(train_set, valAttrX, categories=categories)
    df_res = pd.DataFrame(testAttrX)
    df_res.columns = features + df_res.columns.tolist()[len(features):]

    aso_model = ASOdeep(feature_num=trainAttrX.shape[1], seq_shape=(trainX.shape[1], trainX.shape[2]))
    ens_model = aso_model.create_combined_model(filters=(128, 64, 32, 16), dropout_rate=0.25)

    # tf.keras.utils.plot_model(ens_model, "my_first_model_with_shape_info.png", show_shapes=True)
    ens_model.compile(loss=BinaryCrossentropy(),
                      optimizer='adam',
                      metrics=['accuracy'])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=0.001, patience=25, mode='min',
                                                  restore_best_weights=True)

    history = ens_model.fit([trainAttrX, trainX], train_label,
                            validation_data=([valAttrX, valX], val_label),
                            batch_size=64, epochs=500, verbose=0, callbacks=early_stop)

    # ens_model.save("C:/Users/yagao/Documents/ASO/data/EZH2.keras")

    preds = ens_model.predict([testAttrX, testX])
    # plot_history(history)
    # print(g)
    # test_set["Prediction"] = np.array(preds[:, 1])
    # test_set.to_csv('C:/Users/yagao/Documents/ASO/testgenes/EZH2_test.csv', index=False)

    plot_AUC_ROC(preds, test_label, False)
    plot_AUC_PR(preds, test_label, False)
    # plot_predictions(test_eff, preds)
    roc_values = []
    pr_values = []
    pred_values = []
    probs = np.array(preds[:, 1])
    y_test = np.array(test_label[:, 1])
    fpr, tpr, threshold = roc_curve(y_test, probs)
    precision, recall, _ = precision_recall_curve(y_test, probs)
    for i in range(len(fpr)):
        roc_values.append(['ASOmodel', fpr[i], tpr[i]])
    for i in range(len(precision)):
        pr_values.append(['ASOmodel', precision[i], recall[i]])
    for i in range(len(probs)):
        pred_values.append([y_test[i], probs[i]])
    roc_df = pd.DataFrame(roc_values, columns=['Model', 'FPR', 'TPR'])
    pr_df = pd.DataFrame(pr_values, columns=['Model', 'Precision', 'Recall'])
    pred_df = pd.DataFrame(pred_values, columns=['Actual', 'Predicted'])

    # Write DataFrames to CSV files
    # roc_df.to_csv('ASO_roc_values.csv', index=False)
    pr_df.to_csv('C:/Users/yagao/Documents/ASO/data/ASO_pr_values.csv', index=False)
    pred_df.to_csv('C:/Users/yagao/Documents/ASO/data/a.csv', index=False)

if __name__ == "__main__":
    main()
