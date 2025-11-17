import streamlit as st
from typing import Optional
from core.constants import COLLECTION_NAME

class FlowerAppUI:
    """
    Streamlit UI wrapper for Flower Retrieval + LLM suggestion.
    """

    def __init__(self, service):
        """
        :param service: instance of FlowerService or class that has generate_response()
        """
        self.service = service

    def run(self):
        st.title("🌸 Flower Arrangement Assistant")

        # --- Input text query ---
        query: str = st.text_input("Enter your query (e.g., 'pink flower with yellow center'):")

        if not query:
            st.info("Type a query to get flower suggestions.")
            return

        st.write(f"**Your query:** {query}")

        # --- Retrieval + optional image display ---
        with st.spinner("Retrieving images..."):
            response = self._process_query(query)

        # --- Display response ---
        st.markdown("### Suggestions for bouquet arrangement:")
        st.write(response)

    # ------------------------------
    # Internal: process query + retrieval + LLM
    # ------------------------------
    def _process_query(self, user_query: str, n_results: int = 2, show_images: bool = True) -> str:
        # 1. Retrieve images + LLM
        response = self.service.generate_response(
            collection_name=COLLECTION_NAME,
            user_query=user_query,
            n_results=n_results,
            show_images=show_images
        )
        return response
