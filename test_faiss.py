from utils.embedding import FashionEmbedding
from utils.faiss_search import FaissSearch

embedder = FashionEmbedding()

faiss_search = FaissSearch()

query = (
    "A stylish dinner outfit "
    "for a woman with warm undertone "
    "wearing hijab."
)

query_embedding = embedder.encode_text(query)

results = faiss_search.search(
    query_embedding,
    top_k=5
)

for r in results:
    print(r)