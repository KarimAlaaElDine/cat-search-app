import numpy as np
import tensorflow as tf
import os
import argparse
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

load_dotenv("secrets.env")

DEFAULT_MODEL_PATH = 'weights_best_overall.keras'
REPO_NAME = os.getenv("MODEL_REPO_NAME")
HF_USERNAME = os.getenv("HF_USERNAME")
HF_TOKEN = os.getenv("HF_TOKEN")

parser = argparse.ArgumentParser(description='Upload model to HuggingFace')


parser = argparse.ArgumentParser(description="Upload a TensorFlow model to Hugging Face.")
parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to the model (SavedModel or .h5).")

args = parser.parse_args()
model_path = args.model_path  
repo_id = f"{HF_USERNAME}/{REPO_NAME}"

api = HfApi(token = HF_TOKEN)
# api.set_access_token(HF_TOKEN)
create_repo(repo_id, exist_ok=True)

if model_path.endswith((".keras", ".h5")):
    print(f"Uploading Keras model: {model_path}")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=os.path.basename(model_path),
        repo_id=repo_id
    )

    print(f"✅ Model uploaded successfully to: https://huggingface.co/{repo_id}")

else:
    print("❌ Invalid model path. Provide a valid .keras or .h5 file.")
