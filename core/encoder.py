import base64

class ImageEncoder:
    @staticmethod
    def encode_image_to_base64(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
