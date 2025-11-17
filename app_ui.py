import os

from dotenv import load_dotenv

from core.chroma.chroma_launcher import ChromaLauncher
from core.constants import DB_PATH
import chromadb
from chromadb.config import Settings
from repository.flower_dataset_manager import FlowerDatasetManager
from core.constants import DB_PATH,IMAGE_FOLER,COLLECTION_NAME
from service.FlowerAppUI import FlowerAppUI
from service.flower_visionPrompt_service import FlowerVisionPromptService

def main():
    # Start server
    load_dotenv()
    launcher = ChromaLauncher(data_path=DB_PATH, port=8000)
    launcher.start()
    # Connect to server
    client = chromadb.Client(Settings(
        chroma_api_impl="chromadb.api.fastapi.FastAPI",
        chroma_server_host="127.0.0.1",
        chroma_server_http_port=launcher.port,
    ))


    manager = FlowerDatasetManager(
    dataset_folder=IMAGE_FOLER,
    chroma_path=DB_PATH,)
    # Load dataset
    ds = manager.get_dataset()
    print("Number of rows:", ds.num_rows)
    # Save first 500 images
    # manager.save_images(ds, num_images=500)

    # print(f"ds:", ds)
    # #get image at index=100
    # uri_flower = ds["train"][100]["image"]
    # print(f"uri_flower:{uri_flower}")
    # utils.show_image(uri_flower, title="Flower 1")

    #Save image into database
    collection  = manager.add_images_to_collection_from_folder(IMAGE_FOLER,  COLLECTION_NAME)
    print(f"number of item in a collection:{collection}")
    #query data from a collection
    #query_texts = "Rose flower"
    #results= manager.query(COLLECTION_NAME,query_texts,5)
    #print(f"results:{results}\n")
    #print(f"results[uris][0]: {results["uris"][0]}\n")
    #manager.print_results(results,True)

    # Initialize the Vision Prompt Service
    vision_service = FlowerVisionPromptService(manager)

    # Initilize UI wrapper
    app_ui = FlowerAppUI(vision_service)
    app_ui.run()

if __name__ == "__main__":
    main()