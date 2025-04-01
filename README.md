# Cat Similarity Search using FAISS and CNNs

This projects makes use of Vector databses (**FAISS**) and a CNN trained on cat breed classification to create embeddings that are used to find similar looking cats. 

The CNN uses **EfficientNetV2L** as a backbone with a custom head to classify 5 breeds of cats, the final layer is removed and the layer before it is used to create 128D embedding of the cat images. These embeddings are then stored on the vector database to be used for similarity search. The dataset used for training is [Cat's breed Dataset](https://www.kaggle.com/datasets/yapwh1208/cats-breed-dataset) by user YAPWH

Links to [the model](https://huggingface.co/KarimSayed/cat-breed-encoder) and [the database](https://huggingface.co/datasets/KarimSayed/cat-breed-fiass-index)

# Application Structure

This application contains the following folders and files:

 - **app:**  final model's app created using FastAPI
    - **main.py:** the main app file containing the definitions for the end points
  - **scrtipts:** the location of the model and helper files
      - **faiss_loader.py:** used to run the model on the image database and save the embeddings and image refrences to HuggingFace.
      - **model_updater.py:** used to upload the newest version of the model to hugging face to be used by the app.
