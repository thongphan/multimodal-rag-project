from fastapi import APIRouter
from service.rag_pipeline import RAGPipeline
from repository.chroma_retriever import ChromaRetriever
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

router = APIRouter()

@router.post("/rag/query")
def run_rag(query: str):
    # create a chromadb object
    chroma_client = chromadb.PersistentClient(path="./data/chroma.db")
    # instantiate image loader
    image_loader = ImageLoader()

    # instantiate multimodal embedding function
    embedding_function = OpenCLIPEmbeddingFunction()
    # create the collection, - vector database
    flower_collection = chroma_client.get_or_create_collection(
        "flowers_collection",
        embedding_function=embedding_function,
        data_loader=image_loader,
    )
    retriever = ChromaRetriever(flower_collection)
    pipeline = RAGPipeline(retriever)
    ## === Putting it all together ===
    print("Welcome to the flower arrangement service!")
    print("Please enter your query to get some ideas for a bouquet arrangement.")
    query = input("Enter your query: \n")
    # query in vector DB
    response = pipeline.run(query)
    return {"query": query, "response": response}
