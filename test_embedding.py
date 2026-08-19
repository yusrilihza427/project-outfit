from utils.embedding import FashionEmbedding

embedder = FashionEmbedding()

vector = embedder.encode_image(
    "dataset/img0006.jpg"
)

print(vector.shape)