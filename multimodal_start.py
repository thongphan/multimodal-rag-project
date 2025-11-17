import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from matplotlib import pyplot as plt
import json
import numpy as np
import base64
import warnings


# Suppress all warnings
warnings.filterwarnings("ignore")


# create a chromadb object
chroma_client = chromadb.PersistentClient(path="./data/chroma.db")

# instantiate image loader
image_loader = ImageLoader()

# instantiate multimodal embedding function
embedding_function = OpenCLIPEmbeddingFunction()

# create the collection, - vector database
collection = chroma_client.get_or_create_collection(
    "multimodal_collection",
    embedding_function=embedding_function,
    data_loader=image_loader,
)

# add images to the collection add() or update() method
collection.update(
    ids=["0", "1"],
    uris=["./images/lion.jpg", "./images/tiger.jpg"],
    metadatas=[{"category": "animal"}, {"category": "animal"}],  # metadata - optional
)

# check the count of the collection

collection.update(
    ids=[
        "E23",
        "E25",
        "E33",
    ],
    uris=[
        "./images/E23-2.jpg",
        "./images/E25-2.jpg",
        "./images/E33-2.jpg",
    ],
    metadatas=[
        {
            "item_id": "E23",
            "category": "food",
            "item_name": "Braised Fried Tofu with Greens",
        },
        {
            "item_id": "E25",
            "category": "food",
            "item_name": "Sauteed Assorted Vegetables",
        },
        {"item_id": "E33", "category": "food", "item_name": "Kung Pao Tofu"},
    ],
)
#print(collection.count())  # res: 2

# Simple function to print the results of a query.
# The 'results' is a dict {ids, distances, data, ...}
# Each item in the dict is a 2d list.
def print_query_results(query_list: list, query_results: dict) -> None:
    result_count = len(query_results["ids"][0])

    for i in range(len(query_list)):
        print(f"Results for query: {query_list[i]}")

        for j in range(result_count):
            id = query_results["ids"][i][j]
            distance = query_results["distances"][i][j]
            data = query_results["data"][i][j]
            document = query_results["documents"][i][j]
            metadata = query_results["metadatas"][i][j]
            uri = query_results["uris"][i][j]

            print(
                f"id: {id}, distance: {distance}, metadata: {metadata}, document: {document}"
            )

            # Display image, the physical file must exist at URI.
            # (ImageLoader loads the image from file)
            print(f"data: {uri}")
            plt.imshow(data)
            plt.axis("off")
            plt.show()


# It is possible to submit multiple queries at the same time, just add to the list.
# --- Converter function for JSON serialization ---
def to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        # Option 1: compress image data (shape + dtype + base64)
        return {
            "type": "ndarray",
            "shape": obj.shape,
            "dtype": str(obj.dtype),
            "data_base64": base64.b64encode(obj.tobytes()).decode('utf-8')
        }
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_serializable(v) for v in obj]
    else:
        return obj




query_texts = ["food with carrots", "human"]

# Query vector db - return 3 results
query_results = collection.query(
    query_texts=query_texts,
    n_results=2,
    include=["documents", "distances", "metadatas", "data", "uris"],
    # where={"category": "animal"}, # filter by metadata - optional - first run remove this
)
#print(query_results)
#result_count = len(query_results["ids"][0])
#print(result_count)
# --- Convert the whole structure ---
#json_result = to_json_serializable(query_results)

# --- Return or print result as JSON string ---
#json_output = json.dumps(json_result, indent=2)
#print(json_output)
print_query_results(query_texts, query_results)