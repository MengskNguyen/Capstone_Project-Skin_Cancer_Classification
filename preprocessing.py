import joblib
import cv2
import numpy as np
import pandas as pd

localization_cat_encoder = joblib.load("Encoders/localization_cat_encoder.joblib")
sex_cat_label_encoder = joblib.load("Encoders/sex_cat_label_encoder.joblib")
scaler = joblib.load("Scalers/scaler.joblib")
pca = joblib.load("Encoders/pca.joblib")

def data_preprocessing(df):
    df['localization'] = localization_cat_encoder.transform(df['localization'])
    df['sex'] = sex_cat_label_encoder.transform(df['sex'])
    return df


def image_preprocessing(img_input):
    image_size = (32, 32)
    pixels = []
    if isinstance(img_input, str):
        img = cv2.imread(img_input)
    else:
        img = img_input

    img_resized = cv2.resize(img, image_size)
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_flat = img_resized.flatten()
    pixels.append(img_flat)
    pixels = np.array(pixels)
    x_pca = pca.transform(pixels)
    df = pd.DataFrame(x_pca)
    return df

def preprocessing(df, img_input):
    # xử lý data dạng bảng
    data_df = data_preprocessing(df)

    # xử lý ảnh
    if isinstance(img_input, str):  # nếu là path ảnh
        image_df = image_preprocessing(img_input)
    elif isinstance(img_input, np.ndarray):  # nếu là numpy array
        image_df = image_preprocessing(img_input)
    else:
        raise ValueError("img_input phải là đường dẫn (str) hoặc numpy array")

    # merge 2 DataFrame lại
    merged_df = pd.merge(image_df, data_df, left_index=True, right_index=True)
    merged_df.columns = merged_df.columns.astype(str)

    # scale dữ liệu
    final_df = scaler.transform(merged_df)
    return final_df

# def preprocessing(df, img_path):
#     data_df = data_preprocessing(df)
#     image_df = image_preprocessing(img_path)
#     merged_df = pd.merge(image_df, data_df, left_index=True, right_index=True)
#     merged_df.columns = merged_df.columns.astype(str)
#     final_df = scaler.transform(merged_df)
#     return final_df
