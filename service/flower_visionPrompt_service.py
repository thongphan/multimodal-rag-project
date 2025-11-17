import base64
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any
from repository.flower_dataset_manager import FlowerDatasetManager
from utils import utils


class FlowerVisionPromptService:
    """
    SOLID service to handle prompt-based image retrieval and LLM responses.
    SRP: this class only handles retrieval + formatting prompt + calling OpenAI.
    """

    def __init__(self, db_manager: FlowerDatasetManager, model_name: str = "gpt-4o"):
        self.db_manager = db_manager
        self.vision_model = ChatOpenAI(model=model_name, temperature=0.0)
        self.parser = StrOutputParser()

    # ------------------------------
    # Build prompt template
    # ------------------------------
    def _build_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a talented florist and you have been asked to create a bouquet of flowers for a special event. "
                "Answer the user's question using the given image context with direct references to parts of the images provided. "
                "Maintain a conversational tone; avoid long lists. Use markdown formatting for emphasis."
            ),
            (
                "user",
                [
                    {"type": "text", "text": "what are some good ideas for a bouquet arrangement {user_query}"},
                    {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data_1}"},
                    {"type": "image_url", "image_url": "data:image/jpeg;base64,{image_data_2}"},
                ]
            )
        ])

    # ------------------------------
    # Format images for LLM
    # ------------------------------
    @staticmethod
    def format_prompt_inputs(results: Dict[str, Any], user_query: str) -> Dict[str, str]:
        inputs = {"user_query": user_query}
        image_paths = results["uris"][0][:2]  # take first two images
        for idx, path in enumerate(image_paths, start=1):
            with open(path, "rb") as f:
                img_bytes = f.read()
            inputs[f"image_data_{idx}"] = base64.b64encode(img_bytes).decode("utf-8")
        return inputs

    # ------------------------------
    # Main method to run retrieval + LLM
    # ------------------------------
    def generate_response(self,collection_name:str, user_query: str, n_results: int = 2, show_images: bool = True) -> str:
        # 1. Retrieve images from DB
        results = self.db_manager.query(collection_name,user_query, n_results=n_results)

        # 2. Optionally show images
        if show_images:
            for uri in results["uris"][0]:
                utils.show_image_from_uri(uri)

        # 3. Format prompt inputs
        prompt_input = self.format_prompt_inputs(results, user_query)
        print("prompt_input", prompt_input)

        # 4. Build prompt template
        prompt_template = self._build_prompt_template()

        # 5. Run chain
        vision_chain = prompt_template | self.vision_model | self.parser
        print("vision_chain", vision_chain)
        response = vision_chain.invoke(prompt_input)
        return response
