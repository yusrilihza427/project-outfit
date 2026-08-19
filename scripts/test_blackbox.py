import os
import sys
import numpy as np

# =========================================================
# Project root
# =========================================================
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from utils.embedding import FashionEmbedding
from utils.faiss_search import FaissSearch
from utils.rule_based import RuleBasedFilter


# =========================================================
# PATH
# =========================================================
METADATA_PATH = "metadata/dataset_outfit_final_revisi.csv"


# =========================================================
# LOAD COMPONENTS
# =========================================================
print("=" * 65)
print("BLACK BOX TESTING - OUTFIT RECOMMENDATION SYSTEM")
print("=" * 65)

print("\nLoading system components...")

rule_filter = RuleBasedFilter(METADATA_PATH)
embedder = FashionEmbedding()
faiss_search = FaissSearch()

print("\nSystem components loaded successfully.")


# =========================================================
# TEST RESULT STORAGE
# =========================================================
test_results = []


def record_test(no, feature, expected, condition, actual):
    status = "PASS" if condition else "FAIL"

    test_results.append({
        "No": no,
        "Feature": feature,
        "Expected": expected,
        "Actual": actual,
        "Status": status
    })

    print(f"[{status}] {feature}")
    print(f"       Expected : {expected}")
    print(f"       Actual   : {actual}\n")


# =========================================================
# TEST INPUT
# =========================================================
user_input = {
    "aktivitas": "dinner",
    "undertone": "cool",
    "hijab": "ya"
}

print("\n" + "=" * 65)
print("TEST SCENARIO")
print("=" * 65)

print("Activity  :", user_input["aktivitas"])
print("Undertone :", user_input["undertone"])
print("Hijab     :", user_input["hijab"])

print("\n" + "=" * 65)
print("TEST EXECUTION")
print("=" * 65)


# =========================================================
# TEST 1 - ACTIVITY INPUT
# =========================================================
condition = (
    user_input["aktivitas"]
    in RuleBasedFilter.VALID_ACTIVITY
)

record_test(
    1,
    "Activity Input",
    "Valid activity input is accepted",
    condition,
    f"Activity '{user_input['aktivitas']}' accepted"
)


# =========================================================
# TEST 2 - UNDERTONE INPUT
# =========================================================
condition = (
    user_input["undertone"]
    in RuleBasedFilter.VALID_UNDERTONE
)

record_test(
    2,
    "Skin Undertone Input",
    "Valid undertone input is accepted",
    condition,
    f"Undertone '{user_input['undertone']}' accepted"
)


# =========================================================
# TEST 3 - HIJAB INPUT
# =========================================================
condition = (
    user_input["hijab"]
    in RuleBasedFilter.VALID_HIJAB
)

record_test(
    3,
    "Hijab Input",
    "Valid hijab preference is accepted",
    condition,
    f"Hijab '{user_input['hijab']}' accepted"
)


# =========================================================
# RUN RULE-BASED FILTER
# =========================================================
try:
    rule_result = rule_filter.filter(user_input)
    rule_success = True

except Exception as e:
    rule_result = None
    rule_success = False
    rule_error = str(e)


# =========================================================
# TEST 4 - RULE MATCHING
# =========================================================
if rule_success:

    recommended_colors = rule_result.get(
        "recommended_colors",
        []
    )

    condition = len(recommended_colors) > 0

    actual = (
        ", ".join(recommended_colors)
        if recommended_colors
        else "No recommended colors"
    )

else:
    condition = False
    actual = rule_error


record_test(
    4,
    "Rule Matching",
    "System generates recommended colors",
    condition,
    actual
)


# =========================================================
# TEST 5 - METADATA FILTERING
# =========================================================
if rule_success:

    candidate_ids = rule_result["candidate_ids"]
    total_candidate = rule_result["total_candidate"]

    condition = (
        total_candidate > 0
        and len(candidate_ids) == total_candidate
    )

    actual = (
        f"{total_candidate} outfit candidates generated"
    )

else:
    candidate_ids = []
    condition = False
    actual = "Filtering failed"


record_test(
    5,
    "Metadata Filtering",
    "System generates outfit candidate IDs",
    condition,
    actual
)


# =========================================================
# TEST 6 - FASHIONSIGLIP TEXT ENCODING
# =========================================================
try:

    query = rule_result["query"]

    query_embedding = embedder.encode_text(query)

    condition = (
        isinstance(query_embedding, np.ndarray)
        and query_embedding.shape[0] == 768
    )

    actual = (
        f"Query embedding generated "
        f"with dimension {query_embedding.shape}"
    )

except Exception as e:

    condition = False
    actual = str(e)
    query_embedding = None


record_test(
    6,
    "Marqo-FashionSigLIP Text Encoding",
    "Text query is converted into 768-dimensional embedding",
    condition,
    actual
)


# =========================================================
# TEST 7 - FAISS RETRIEVAL
# =========================================================
if query_embedding is not None:

    try:

        results = faiss_search.search(
            query_embedding,
            top_k=5,
            candidate_ids=candidate_ids
        )

        condition = len(results) > 0

        actual = (
            f"{len(results)} ranked outfits retrieved"
        )

    except Exception as e:

        results = []
        condition = False
        actual = str(e)

else:

    results = []
    condition = False
    actual = "Query embedding unavailable"


record_test(
    7,
    "FAISS Retrieval",
    "System retrieves and ranks candidate outfits",
    condition,
    actual
)


# =========================================================
# TEST 8 - TOP 3 RECOMMENDATION
# =========================================================
top3 = results[:3]

condition = len(top3) == 3

if condition:

    actual = (
        f"Top-3 successfully generated: "
        f"{top3[0]['nama_gambar']}, "
        f"{top3[1]['nama_gambar']}, "
        f"{top3[2]['nama_gambar']}"
    )

else:

    actual = (
        f"Only {len(top3)} recommendation(s) generated"
    )


record_test(
    8,
    "Top-3 Recommendation",
    "System returns three highest-ranked outfits",
    condition,
    actual
)


# =========================================================
# DETAIL TOP-3
# =========================================================
print("=" * 65)
print("TOP-3 RECOMMENDATION RESULT")
print("=" * 65)

for rank, result in enumerate(top3, start=1):

    print(
        f"{rank}. "
        f"{result['nama_gambar']} | "
        f"ID: {result['id']} | "
        f"Similarity: {result['score']:.4f}"
    )


# =========================================================
# FINAL SUMMARY
# =========================================================
passed = sum(
    1 for result in test_results
    if result["Status"] == "PASS"
)

failed = len(test_results) - passed


print("\n" + "=" * 65)
print("BLACK BOX TEST SUMMARY")
print("=" * 65)

for result in test_results:

    print(
        f"{result['No']}. "
        f"{result['Feature']:<38} "
        f"{result['Status']}"
    )


print("-" * 65)

print(f"Total Test : {len(test_results)}")
print(f"PASS       : {passed}")
print(f"FAIL       : {failed}")

if failed == 0:
    print("\nFINAL RESULT: ALL FUNCTIONAL TESTS PASSED")
else:
    print("\nFINAL RESULT: SOME FUNCTIONAL TESTS FAILED")

print("=" * 65)