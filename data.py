# import the necessary packages
import math
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, RobustScaler
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from itertools import product
from collections import Counter
import joblib
from Bio import SeqIO


def one_hot_encode(seq, max_length=20):
    encode = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    x = [encode[x] for x in seq]
    if len(x) < max_length:
        delta = max_length - len(x)
        left_padding = math.floor(delta * 0.5)
        right_padding = delta - left_padding
        x = [[0, 0, 0, 0]] * left_padding + x + [[0, 0, 0, 0]] * right_padding
    return x


def kmers_encode(sequences, k=3):
    dna_alphabet = ['A', 'C', 'G', 'T']
    all_kmers = ["".join(x) for x in product(dna_alphabet, repeat=k)]
    kmer_matrix = []
    for seq in sequences:
        counts = Counter([seq[i:i + 3] for i in range(len(seq) - 2)])
        vector = [counts[mer] if mer in counts else 0 for mer in all_kmers]
        kmer_matrix.append(vector)

    return np.array(kmer_matrix)


class DataReader:
    def __init__(self, inputPath):
        self.inputPath = inputPath
        self.df = pd.read_csv(self.inputPath)

    def load_train_set(self, encoding='one_hot', max_length=20, k=3):
        sequences = list(self.df.loc[:, 'Sequence'])

        encoded_sequences = []
        for seq in sequences:
            if encoding == 'one_hot':
                x = one_hot_encode(seq, max_length)
            elif encoding == 'kmer':
                x = kmers_encode(seq, k)
            else:
                raise ValueError(f'Unknown encoding: {encoding}')
            encoded_sequences.append(x)
        encoded_sequences = np.array(encoded_sequences)
        labels = self.df["Classify"]
        labels = tf.keras.utils.to_categorical(labels)
        efficacy = self.df["Inhibition"]

        return self.df, encoded_sequences, labels, efficacy

    def load_predict_set(self, encoding='one_hot', max_length=20, k=3):
        sequences = list(self.df.loc[:, 'Sequence'])

        encoded_sequences = []
        for seq in sequences:
            if encoding == 'one_hot':
                x = one_hot_encode(seq, max_length)
            elif encoding == 'kmer':
                x = kmers_encode(seq, k)
            else:
                raise ValueError(f'Unknown encoding: {encoding}')
            encoded_sequences.append(x)
        encoded_sequences = np.array(encoded_sequences)

        return self.df, encoded_sequences


class NoneSeqFeatureProcessor:
    def __init__(self, continuous, category, plain):
        self.cs = StandardScaler()
        self.le = LabelEncoder()
        self.continuous = continuous
        self.category = category
        self.plain = plain

    def process_train_features(self, train, test, categories):
        trainX_continuous = self.cs.fit_transform(train[self.continuous]) if self.continuous else np.array([])
        testX_continuous = self.cs.transform(test[self.continuous]) if self.continuous else np.array([])
        if self.category:
            self.le.fit(np.array(categories).ravel())
            trainX_category = to_categorical(self.le.transform(np.array(train[self.category]).ravel()),
                                             num_classes=len(self.le.classes_))
            testX_category = to_categorical(self.le.transform(np.array(test[self.category]).ravel()),
                                            num_classes=len(self.le.classes_))
        else:
            trainX_category = testX_category = np.array([])

        joblib.dump([self.cs, self.le], 'C:/Users/yagao/Documents/ASO/data/ASOscalers.joblib')
        if self.plain:
            if len(self.plain) != 1 :
                train_plain = train[self.plain].values
                test_plain = test[self.plain].values
            else:
                train_plain = train[self.plain].values.reshape(-1, 1)
                test_plain = test[self.plain].values.reshape(-1, 1)
        else:
            train_plain = test_plain = np.array([])

        trainX = np.hstack((trainX_continuous, trainX_category, train_plain))
        testX = np.hstack((testX_continuous, testX_category, test_plain))
        return (trainX, testX)

    def process_predict_features(self, df, scaler_path):
        cs, le = joblib.load(scaler_path)
        pred_continous = cs.transform(df[self.continuous]) if self.continuous else np.array([])
        if self.category:
            pred_category = to_categorical(le.transform(np.array(df[self.category]).ravel()), num_classes=len(le.classes_))
        else:
            pred_category = np.array([])
        if self.plain:
            if len(self.plain) != 1 :
                pred_plain = df[self.plain].values
            else:
                pred_plain = df[self.plain].values.reshape(-1, 1)
        else:
            pred_plain = np.array([])

        pred = np.hstack((pred_continous, pred_category,pred_plain))

        return pred


class SingleFeatureProcessor:
    def __init__(self, feature_type, feature_name):
        self.cs = StandardScaler()
        self.le = LabelEncoder()
        self.feature_type = feature_type
        self.feature_name = feature_name

    def process_feature(self, train, test, categories=None):
        if self.feature_type == 'continuous':
            trainX = self.cs.fit_transform(train[[self.feature_name]])
            testX = self.cs.transform(test[[self.feature_name]])
        elif self.feature_type == 'category':
            self.le.fit(np.array(categories).ravel())
            trainX = to_categorical(self.le.transform(np.array(train[[self.feature_name]]).ravel()),
                                    num_classes=len(self.le.classes_))
            testX = to_categorical(self.le.transform(np.array(test[[self.feature_name]]).ravel()),
                                   num_classes=len(self.le.classes_))
        elif self.feature_type == 'plain':
            trainX = train[[self.feature_name]].values
            testX = test[[self.feature_name]].values
        else:
            raise ValueError("Invalid feature type. Choose from 'continuous', 'category', or 'plain'.")

        joblib.dump([self.cs, self.le], 'C:/Users/yagao/Documents/ASO/data/ASOscalers.joblib')

        return (trainX, testX)