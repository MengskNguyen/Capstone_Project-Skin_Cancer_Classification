import streamlit as st
from PIL import Image
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocessing
import joblib
from matplotlib.patches import Patch


# Tiêu đề
st.title("Capstone Project - Skin Cancer Image Prediction")
st.subheader("Author: Thế Anh Nguyễn")
# Text
st.write("Hello! This is Skin Cancer Image Prediction, using SVM model")

# Slider
age = st.slider("Age:", min_value=0, max_value=100, value=30, step=1)
gender = st.selectbox(
    "Gender:",
    ["male", "female"]
)
localization = st.selectbox(
    "Localization:",
    ['scalp', 'ear', 'face', 'back', 'trunk', 'chest',
    'upper extremity', 'abdomen', 'unknown', 'lower extremity',
    'genital', 'neck', 'hand', 'foot', 'acral']
)

# Upload file
IMAGE_FOLDER = "Datasets/skin-cancer-mnist-ham10000/HAM10000_images_part_1"

# Upload ảnh
uploaded_file = st.file_uploader(
    "Upload image",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False
)

image_path = ""

if uploaded_file:
    # đọc ảnh từ file upload
    image = Image.open(uploaded_file)

    # show ảnh
    st.image(image, caption="Uploaded image", use_container_width=True)

    image_np = np.array(image)

else:
    all_images = os.listdir(IMAGE_FOLDER)

    st.write("Or choose an image:")

    choice = st.selectbox("Choose an image:", all_images)

    image_path = os.path.join(IMAGE_FOLDER, choice)
    st.write(image_path)

    image = Image.open(image_path)
    st.image(image, caption=f"Chosen image: {choice}", use_container_width=True)

if st.button("Predict"):
    # Load models and encoder to inverse later
    binary_clf_model = joblib.load("Models/binary_clf_model.pkl")
    multiclass_clf_model = joblib.load("Models/multi_class_clf_model.pkl")
    diagnose_cat_label_encoder = joblib.load("Encoders/diagnose_cat_label_encoder.joblib")

    df = pd.DataFrame([[age, gender, localization]], columns=["age", "sex", "localization"])
    if(uploaded_file):
        preprocessed_df = preprocessing(df, image_np)
    else:
        preprocessed_df = preprocessing(df, image_path)

    # Predict result bases on given data
    binary_prediction = binary_clf_model.predict(preprocessed_df)
    multiclass_prediction = diagnose_cat_label_encoder.inverse_transform(multiclass_clf_model.predict(preprocessed_df))
    scores = multiclass_clf_model.decision_function(preprocessed_df)

    # Write results

    cancer_classes = ['Melanoma', 'Basal cell carcinoma', 'Actinic keratoses', 'Vascular lesions']

    st.subheader(f"Skin Cancer: {'Positive' if binary_prediction == 1 and multiclass_prediction[0] in cancer_classes else 'Negative'}")
    st.subheader(f"Type: {multiclass_prediction[0]}")

    all_labels = diagnose_cat_label_encoder.classes_

    result = dict(zip(all_labels, scores[0]))
    df_scores = pd.DataFrame(list(result.items()), columns=["Class", "Score"])

    # Plot multiclass prediction

    colors = ["red" if cls in cancer_classes else "skyblue" for cls in df_scores["Class"]]

    # Vẽ bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df_scores["Class"], df_scores["Score"], color=colors)

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"{yval:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Decision Function Score")
    plt.title("SVM Decision Function per Class")

    legend_elements = [
        Patch(facecolor="red", label="Cancer-related"),
        Patch(facecolor="skyblue", label="Non-cancer")
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    st.pyplot(fig)