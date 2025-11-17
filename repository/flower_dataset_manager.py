import os
import streamlit as st
from PIL import Image

from repository.chroma_client import ChromaClient


@st.cache_data
def load_flower_dataset(dataset_name="huggan/flowers-102-categories"):
    from datasets import load_dataset
    return load_dataset(dataset_name)

class FlowerDatasetManager:
    """
    A generic class to manage Flower dataset, images, and ChromaDB.
    Responsibilities:
    - Load dataset from Hugging Face
    - Save images locally
    - Display images
    - Initialize Chroma client
    """

    def __init__(self, dataset_name: str = "huggan/flowers-102-categories",
                 dataset_folder: str = ".\\data",
                 chroma_path: str = ".\\data\\chroma.db"):
        self.dataset_name = dataset_name
        self.dataset_folder = dataset_folder
        self.chroma_path = chroma_path
        self.chroma_client = ChromaClient()

        # Ensure data folder exists
        os.makedirs(self.dataset_folder, exist_ok=True)

    def get_dataset(self):
        return load_flower_dataset(self.dataset_name)
    # ------------------------------
    # Save images from dataset to folder
    # ------------------------------
    def save_images(self, dataset, num_images=500):
        for i in range(num_images):
            print(f"Saving image {i+1} of {num_images}")
            image = dataset["train"][i]["image"]
            image.save(os.path.join(self.dataset_folder, f"flower_{i+1}.png"))

        print(f"Saved {num_images} images to {self.dataset_folder}")
        # ------------------------------
        # Add images from folder to DB
        # ------------------------------

    def add_images_to_collection_from_folder(self, folder_path: str, collection_name:str="flowers_collection"):
        ids, uris = [], []

        # Collect ids and uris
        for i, filename in enumerate(sorted(os.listdir(folder_path))):
            if filename.lower().endswith(".png"):
                file_path = os.path.join(folder_path, filename)
                ids.append(str(i))
                uris.append(file_path)

        flower_collection  = self.chroma_client.get_or_create_collection(collection_name)
        print(f"collection:{flower_collection}")
        flower_collection.add(ids=ids, uris=uris)
        print("Images added to the database.")
        #Validate the VectorDB with .count()
        print(f"Added {len(ids)} images to collection '{flower_collection.count()}'")
        return flower_collection
        # ------------------------------
        # Query DB by text
        # ------------------------------

    def query(self,collection_name:str, query_text: str, n_results: int = 5):
        results = self.chroma_client.get_or_create_collection(collection_name).query(
            query_texts=[query_text],
            n_results=n_results,
            include=["uris", "distances"]
        )
        return results

        # ------------------------------
        # Print results and optionally show images
        # ------------------------------

    @staticmethod
    def print_results(results, show_images=False):
        for idx, uri in enumerate(results["uris"][0]):
            print(f"ID: {results['ids'][0][idx]}")
            print(f"Distance: {results['distances'][0][idx]}")
            print(f"Path: {uri}")
            if show_images and os.path.exists(uri):
                Image.open(uri).show()
            print("\n")

