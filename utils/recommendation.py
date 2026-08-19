from utils.rule_based import RuleBasedFilter
from utils.embedding import FashionEmbedding
from utils.faiss_search import FaissSearch
import time

class RecommendationEngine:

    def __init__(
        self,
        metadata_path="metadata/dataset_outfit_final_revisi.csv"
    ):

        self.rule_based = RuleBasedFilter(metadata_path)
        self.embedder = FashionEmbedding()
        self.faiss = FaissSearch()

    def recommend(
        self,
        user_input,
        top_k=5
    ):

        total_start = time.time()

        # ==========================
        # Rule-Based
        # ==========================
        rb_start = time.time()

        rb_result = self.rule_based.filter(user_input)

        print(f"Candidate : {rb_result['total_candidate']}")
        print(f"Rule-Based : {time.time() - rb_start:.2f} sec")

        # ==========================
        # Text Embedding
        # ==========================
        embed_start = time.time()

        query_embedding = self.embedder.encode_text(
            rb_result["query"]
        )

        print(f"Embedding : {time.time() - embed_start:.2f} sec")

        # ==========================
        # FAISS Search
        # ==========================
        faiss_start = time.time()

        results = self.faiss.search(
            query_embedding=query_embedding,
            candidate_ids=rb_result["candidate_ids"],
            top_k=top_k
        )

        print(f"FAISS : {time.time() - faiss_start:.2f} sec")

        print(f"TOTAL : {time.time() - total_start:.2f} sec")

        return results