import torch
import open_clip
import numpy as np
from PIL import Image


class FashionEmbedding:
    def __init__(self):
        """
        Load FashionSigLIP model sekali saja.
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
            "hf-hub:Marqo/marqo-fashionSigLIP"
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(
            "hf-hub:Marqo/marqo-fashionSigLIP"
        )

        print("✅ FashionSigLIP Loaded")

    def encode_image(self, image_path):
        """
        Encode satu gambar menjadi embedding.
        """

        image = self.preprocess(
            Image.open(image_path).convert("RGB")
        ).unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(
                image,
                normalize=True
            )

        return embedding.cpu().numpy().flatten()

    def encode_text(self, text):
        """
        Encode query text menjadi embedding.
        """

        tokens = self.tokenizer([text]).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_text(
                tokens,
                normalize=True
            )

        return embedding.cpu().numpy().flatten()

    def batch_encode(self, image_paths):
        """
        Encode banyak gambar.
        """

        embeddings = []

        for path in image_paths:
            embeddings.append(
                self.encode_image(path)
            )

        return np.array(embeddings)