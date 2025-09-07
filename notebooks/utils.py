import os
import joblib
import kagglehub
import shutil

def save_to_csv(df, filename, folder="csv"):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    if os.path.exists(file_path):
        print(f"File '{file_path}' is existed")
    else:
        df.to_csv(file_path, index=False)
        print(f"saved {filename} at '{file_path}'.")


def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        print(f"Model is existed at {path}")
    else:
        joblib.dump(model, path)
        print(f"Save Model at {path}")

def save_encoder(encoder, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        print(f"Model is existed at {path}")
    else:
        joblib.dump(encoder, path)
        print(f"Save encoder at {path}")

def save_scaler(scaler, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        print(f"Model is existed at {path}")
    else:
        joblib.dump(scaler, path)
        print(f"Save scaler at {path}")

def download_data():
    path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    print("Cache path:", path)

    target = "../Datasets/skin-cancer-mnist-ham10000"
    os.makedirs(os.path.dirname(target), exist_ok=True)

    if not os.path.exists(target):
        shutil.copytree(path, target)
        print("Copied dataset to:", target)
    else:
        print("Dataset already exists at:", target)