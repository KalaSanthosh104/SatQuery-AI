from PIL import Image

def preprocess_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = cdvqa_transform(
        image
    )

    return image_tensor