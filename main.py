from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
import io
import dotenv
import os
import keras
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import Model 
import faiss
from PIL import Image

app = FastAPI()

# Getting Model from repo
os.environ["KERAS_BACKEND"] = "tensorflow"	
model_path = hf_hub_download(repo_id="KarimSayed/cat-breed-encoder", filename="weights_best_overall.keras")

# Removing layers to use as encoder
encoder = keras.models.load_model(model_path)
removed_layers= 1
encoder = Model(inputs=encoder.input, outputs=encoder.layers[-(removed_layers + 1)].output)

# Getting Vector DB from repo
db_repo = "KarimSayed/cat-breed-fiass-index"

faiss_  = hf_hub_download(repo_id=db_repo, filename="cat_vectors.index", repo_type='dataset')
faiss_db  = faiss.read_index(faiss_)
filenames_  = hf_hub_download(repo_id=db_repo, filename="image_filenames.npy", repo_type='dataset')
image_filenames = np.load(filenames_)

# View the first few filenames
print(image_filenames[:5])
def get_top_db(value=5):
    stored_vectors = np.vstack([faiss_db.reconstruct(i) for i in range(value)])
    return stored_vectors.tolist()

def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_str = stream.getvalue()
    stream.close()
    return summary_str

def preprocess_image(image, target_size=(128, 128)):
    """
    Preprocess the image: Resize, normalize, and convert to numpy array.
    """
    image = image.resize(target_size)  # Resize to 128x128
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image





@app.get("/")
async def root():
    return {"message": "hi"}

@app.get("/db")
async def view_faiss():
    return {"message": get_top_db()}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image and get a prediction from the model.
    """
    try:
        # Read image
        image = Image.open(io.BytesIO(await file.read()))
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Get prediction
        query_vector = encoder.predict(processed_image).flatten()

        k = 5  # Number of nearest neighbors to retrieve
        
        D, I = faiss_db.search(np.array([query_vector]), k)
        
        d = faiss_db.d  # Dimensionality

        # Perform a search
        matching_filenames = image_filenames[I[0]].tolist()
        matching_filenames = ["".join(('images/', x.split('/')[-1])) for x in matching_filenames]
        
        return {"Filenames": matching_filenames}
    
    except Exception as e:
        return {"error": str(e)}