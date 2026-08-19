import os
import sys

# ==================================================
# Agar Python menemukan project root
# ==================================================
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from utils.rule_based import RuleBasedFilter
from utils.embedding import FashionEmbedding
from utils.faiss_search import FaissSearch


# ==================================================
# LOAD COMPONENTS
# ==================================================
print("\n========================================")
print("LOADING SYSTEM")
print("========================================")

rule_based = RuleBasedFilter(
    "metadata/dataset_outfit_final_revisi.csv"
)

embedder = FashionEmbedding()

faiss_search = FaissSearch()


# ==================================================
# USER INPUT
# ==================================================
user_input = {
    "aktivitas": "dinner",
    "undertone": "cool",
    "hijab": "ya"
}


print("\n========================================")
print("USER INPUT")
print("========================================")

print("Aktivitas :", user_input["aktivitas"])
print("Undertone :", user_input["undertone"])
print("Hijab     :", user_input["hijab"])


# ==================================================
# STEP 1 — RULE-BASED SYSTEM
# ==================================================
rule_result = rule_based.filter(
    user_input
)


print("\n========================================")
print("RULE-BASED RESULT")
print("========================================")

print("Recommended Colors:")

for color in rule_result["recommended_colors"]:
    print(
        "-",
        color.replace("_", " ").title()
    )

print(
    "\nTotal Candidate :",
    rule_result["total_candidate"]
)


# ==================================================
# STEP 2 — QUERY FASHIONSIGLIP
# ==================================================
query = rule_result["query"]

print("\n========================================")
print("FASHIONSIGLIP QUERY")
print("========================================")

print(query)


# ==================================================
# STEP 3 — TEXT EMBEDDING
# ==================================================
query_embedding = embedder.encode_text(
    query
)

print("\n========================================")
print("QUERY EMBEDDING")
print("========================================")

print(
    "Embedding Dimension :",
    query_embedding.shape
)


# ==================================================
# STEP 4 — FAISS SEARCH
# ==================================================
results = faiss_search.search(
    query_embedding,
    top_k=5,
    candidate_ids=rule_result["candidate_ids"]
)


# ==================================================
# STEP 5 — DISPLAY RESULTS
# ==================================================
print("\n========================================")
print("FAISS TOP-5 RESULTS")
print("========================================")

if len(results) == 0:

    print("Tidak ada kandidat ditemukan.")

else:

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{i}. "
            f"{result['nama_gambar']} | "
            f"ID: {result['id']} | "
            f"Similarity: {result['score']:.4f}"
        )

print("\n========================================")
print("VALIDATION OF FAISS RESULTS")
print("========================================")

candidate_df = rule_result["candidate_df"]

for i, result in enumerate(results, start=1):

    matched = candidate_df[
        candidate_df["id"].astype(int)
        == result["id"]
    ]

    if not matched.empty:

        row = matched.iloc[0]

        print(
            f"{i}. {result['nama_gambar']} | "
            f"Activity: {row['aktivitas']} | "
            f"Undertone: {row['undertone']} | "
            f"Hijab: {row['hijab']} | "
            f"Color: {row['warna_dominan']} | "
            f"VALID ✓"
        )

    else:

        print(
            f"{i}. {result['nama_gambar']} | "
            f"INVALID ❌"
        )

print("\n========================================")
print("INTEGRATION TEST COMPLETED")
print("========================================")