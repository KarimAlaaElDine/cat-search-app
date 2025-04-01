import kagglehub
import os
import faiss
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import HfApi
from tensorflow.keras.models import Model

# 1️⃣ Download dataset from Kaggle
print("Downloading dataset...")
dataset_path = kagglehub.dataset_download("yapwh1208/cats-breed-dataset")
image_dir = os.path.join(dataset_path, "cat_v1")  # Adjust if necessary

# 2️⃣ Load the encoder model
print("Loading encoder model...")
encoder = tf.keras.models.load_model("weights_best_overall.keras")  # Ensure this path is correct
removed_layers= 1

encoder = Model(inputs=encoder.input, outputs=encoder.layers[-(removed_layers + 1)].output)
# Process images & extract feature vectors
print("Processing images...")
image_filenames = []
feature_vectors = []
image_labels = []

# Loop through each subfolder (each breed folder) to capture images
for breed_folder in os.listdir(image_dir):
    breed_folder_path = os.path.join(image_dir, breed_folder)

    if os.path.isdir(breed_folder_path):  # Check if it's a folder
        for img_file in os.listdir(breed_folder_path):
            img_path = os.path.join(breed_folder_path, img_file)

            img = Image.open(img_path).resize((128, 128))
            img = img.convert('RGB')
            img_array = np.array(img) / 255.0
            img_tensor = np.expand_dims(img_array, axis=0)

            # Get the feature vector for this image
            feature_vector = encoder.predict(img_tensor)

            feature_vectors.append(feature_vector)
            image_filenames.append(img_path)  # Save full path
            image_labels.append(breed_folder)  # Store the breed label

# Convert feature vectors to numpy array
feature_vectors = np.vstack(feature_vectors)

# Build FAISS Index
print(f"🔍 Building FAISS index... size {feature_vectors.shape[1]}")
index = faiss.IndexFlatL2(feature_vectors.shape[1])  # Use the correct dimensionality from feature_vectors
print(feature_vectors.shape[1])
index.add(feature_vectors)

# Save FAISS Index & Image Filenames
os.makedirs("faiss_index", exist_ok=True)
faiss_file = "faiss_index/cat_vectors.index"
np.save("faiss_index/image_filenames.npy", np.array(image_filenames))  # Save filenames

faiss.write_index(index, faiss_file)
print(f"FAISS index created")

# Push to Hugging Face (with folder structure)
print("Uploading to Hugging Face Hub...")
hf_api = HfApi()
repo_id = "KarimSayed/cat-breed-fiass-index"  # Change to your repo
repo_type="dataset"
# Upload the FAISS index and filenames
hf_api.upload_file(
    path_or_fileobj=faiss_file,
    path_in_repo="cat_vectors.index",
    repo_id=repo_id,
    repo_type=repo_type
)

hf_api.upload_file(
    path_or_fileobj="faiss_index/image_filenames.npy",
    path_in_repo="image_filenames.npy",
    repo_id=repo_id,
    repo_type=repo_type
)

# # Now, we need to upload the images while maintaining the folder structure
# for breed_folder in os.listdir(image_dir):
#     breed_folder_path = os.path.join(image_dir, breed_folder)

#     if os.path.isdir(breed_folder_path):  # Check if it's a folder
#         for img_file in os.listdir(breed_folder_path):
#             img_path = os.path.join(breed_folder_path, img_file)

#             # Upload the image
#             hf_api.upload_file(
#                 path_or_fileobj=img_path,
#                 path_in_repo=f"images/{img_file}",
#                 repo_id=repo_id,
#                 repo_type=repo_type
#             )

print(f"✅ FAISS index, filenames, and images uploaded to Hugging Face: {repo_id}")
