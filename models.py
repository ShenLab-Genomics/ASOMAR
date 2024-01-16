from keras.models import Model, Sequential
from keras.layers import Conv1D, Dropout, BatchNormalization, MaxPooling1D, Flatten
from keras.layers import Bidirectional, GRU, LSTM
from keras.layers import concatenate, Dense, Input


class ASOdeep:
    def __init__(self, feature_num, seq_shape):
        self.input_layer = None
        self.units = None
        self.cnn_input = Input(shape=seq_shape)
        self.mlp_input = Input(shape=(feature_num,))

    def create_mlp(self):
        mlp_model = Dense(8, activation="relu")(self.mlp_input)
        mlp_model = Dense(4, activation='relu')(mlp_model)
        return mlp_model

    def create_cnn1d(self, filters, dropout_rate):
        cnn_model = Conv1D(filters[0], kernel_size=4, activation='relu')(self.cnn_input)
        cnn_model = Dropout(dropout_rate)(cnn_model)
        cnn_model = BatchNormalization()(cnn_model)
        for i in range(len(filters)-1):
            cnn_model = Conv1D(filters[i+1], kernel_size=4, activation='relu')(cnn_model)
            cnn_model = Dropout(dropout_rate)(cnn_model)
            cnn_model = BatchNormalization()(cnn_model)
        cnn_model = MaxPooling1D(pool_size=2)(cnn_model)
        cnn_model = Flatten()(cnn_model)

        return cnn_model

    def create_lstm(self):
        lstm_model = LSTM(units=self.units)(self.input_layer)
        lstm_model = Dropout(0.5)(lstm_model)

        return lstm_model

    def create_combined_model(self, filters=(64, 32), dropout_rate=0.25):
        cnn_model = self.create_cnn1d(filters=filters, dropout_rate=dropout_rate)
        nsf_model = self.create_mlp()

        combinedInput = concatenate([cnn_model, nsf_model], axis=1)

        x = Dropout(0.25)(combinedInput)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.25)(x)
        cnn_out = Dense(2, activation='softmax')(x)

        return Model(inputs=[self.mlp_input, self.cnn_input], outputs=cnn_out)

    def seq_only_model(self, filters=(64,32), dropout_rate=0.25):
        cnn_model = self.create_cnn1d(filters=filters, dropout_rate=dropout_rate)
        x = Dropout(0.25)(cnn_model)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(16, activation='relu')(x)
        x = Dropout(0.25)(x)
        cnn_out = Dense(2, activation='sigmoid')(x)

        return Model(inputs=[self.cnn_input], outputs=cnn_out)