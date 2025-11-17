class ImageController:
    def __init__(self, service):
        self.service = service

    def add_image(self, url: str):
        path = self.service.download_image(url)
        print(f"Image stored at: {path}")
        return path