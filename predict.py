from data import DataReader, NoneSeqFeatureProcessor
import tensorflow as tf
import numpy as np


def main():
    model = tf.keras.models.load_model("C:/Users/yagao/Documents/ASO/data/EZH2.keras")
    predict_data = DataReader("C:/Users/yagao/Documents/ASO/testgenes/ZBP1.csv")
    test_df, test_seq= predict_data.load_predict_set()
    features = ["concentration", "self_bind", "open_prob", "dG", "MFE", "ASOMFE", "TM"]
    category = ["modify"]
    plain = ["max_open_length", "open_pc"]
    feature_processor = NoneSeqFeatureProcessor(continuous=features, category=category, plain=plain)
    scaler_path = "C:/Users/yagao/Documents/ASO/data/ASOscalers.joblib"
    test_attr = feature_processor.process_predict_features(df=test_df, scaler_path=scaler_path)

    prediction = model.predict([test_attr, test_seq])
    test_df["Class.1.Probability"] = np.array(prediction[:, 1])
    test_df['Prediction'] = np.where(np.array(prediction[:, 1]) < 0.5, 0, 1)
    # Write DataFrame to CSV
    test_df.to_csv('C:/Users/yagao/Documents/ASO/testgenes/ZBP1_res.csv', index=False)


if __name__ == "__main__":
    main()
