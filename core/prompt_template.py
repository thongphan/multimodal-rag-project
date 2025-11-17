from langchain_core.prompts import ChatPromptTemplate

class PromptFactory:
    @staticmethod
    def florist_prompt():
        return ChatPromptTemplate.from_template("""
        System: You are a talented florist helping users design bouquets.
        Use provided images and context to give meaningful, creative recommendations.
        User Query: {user_query}
        Image 1 (Base64): {image_data1}
        Image 2 (Base64): {image_data2}
        """)

    # Define the prompt template
    @staticmethod
    def image_prompt():
            image_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a talented florist and you have been asked to create a bouquet of flowers for a special event. Answer the user's question  using the given image context with direct references to parts of the images provided."
                    " Maintain a more conversational tone, don't make too many lists. Use markdown formatting for highlights, emphasis, and structure.",
                ),
                (
                    "user",
                    [
                        {
                            "type": "text",
                            "text": "what are some good ideas for a bouquet arrangement {user_query}",
                        },
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_data_1}",
                        },
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_data_2}",
                        },
                    ],
                ),
            ]
        )
            return image_prompt
