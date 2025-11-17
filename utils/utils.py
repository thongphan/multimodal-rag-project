import os
from pathlib import Path
from matplotlib import pyplot as plt
from PIL import Image
def ensure_folder(path: str):
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder
def show_image(uri: str, title: str = None):
    """
    Display an image from a file path.

    Args:
        uri (str): Path to the image file.
        title (str, optional): Title to display above the image.
    """
    try:
        plt.imshow(uri)
        plt.title(title)
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"❌ Failed to display image {uri}: {e}")

def show_image_from_uri(uri: str):
    """
    Open and display an image from a file path using PIL only.

    Args:
        uri (str): Path to the image file
    """
    if not os.path.exists(uri):
        print(f"❌ Image file does not exist: {uri}")
        return

    try:
        img = Image.open(uri)
        img.show()  # opens default image viewer
    except Exception as e:
        print(f"❌ Failed to show image {uri}: {e}")