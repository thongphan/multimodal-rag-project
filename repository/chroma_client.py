# repository/chroma_client.py

import os
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from typing import Any
from chromadb import HttpClient

class ChromaClient:
    """Manages the persistent connection to the local ChromaDB instance."""
    def __init__(self):
        self.chroma_client = HttpClient(host="localhost", port=8000)
        # instantiate image loader
        self.image_loader = ImageLoader()

        # instantiate multimodal embedding function
        self.embedding_function = OpenCLIPEmbeddingFunction()
        print("ChromaDB Client initialized successfully for local persistence.")

    def get_client(self) -> Any:
        """Returns the initialized ChromaDB client."""
        return self.chroma_client

    # create the collection, - vector database
    def get_or_create_collection(self, collection_name:str) -> Any:
        """Helper to call get_or_create_collection on the underlying client."""
        return self.chroma_client.get_or_create_collection(collection_name, embedding_function=self.embedding_function,data_loader=self.image_loader)