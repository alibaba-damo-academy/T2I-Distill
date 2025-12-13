
from torch.utils.data import Dataset
import torchvision.transforms as T


class Text2ImageDataset(Dataset):
    """
    A PyTorch Dataset class for loading text-image pairs from a HuggingFace dataset.
    This dataset is designed for text-to-image generation tasks.
    Args:
        hf_dataset (datasets.Dataset):
            A HuggingFace dataset containing 'image' (bytes) and 'llava' (text) fields. Note that 'llava' is the field name for text descriptions in this specific dataset - you may need to adjust this key if using a different HuggingFace dataset with a different text field name.
            resolution (int, optional): Target resolution for image resizing. Defaults to 1024.
    Returns:
        dict: A dictionary containing:
            - 'text': The text description (str)
            - 'image': The processed image tensor (torch.Tensor) of shape [3, resolution, resolution]
    """

    def __init__(self, hf_dataset, resolution=1024):
        self.dataset = hf_dataset
        self.transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB")),
                T.Resize(resolution),  # Image.BICUBIC
                T.CenterCrop(resolution),
                T.ToTensor(),
                T.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["llava"]
        image_bytes = item["image"]

        # Convert bytes to PIL Image
        # image = Image.open(io.BytesIO(image_bytes))  # puyifan: image_bytes is already <class 'PIL.JpegImagePlugin.JpegImageFile'>
        image = image_bytes

        image_tensor = self.transform(image)

        # return {"text": text, "image": image_tensor}
        return image_tensor, text

