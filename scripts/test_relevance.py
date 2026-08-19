import os
import sys
import itertools
import pandas as pd

# =========================================================
# PROJECT ROOT
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
OUTPUT_PATH = "evaluation/relevance_test_results.csv"


# =========================================================
# LOAD COMPONENTS
# =========================================================
print("=" * 80)
print("RELEVANCE@3 TEST - ALL INPUT COMBINATIONS")
print("=" * 80)

print("\nLoading system components...")

rule_filter = RuleBasedFilter(METADATA_PATH)
embedder = FashionEmbedding()
faiss_search = FaissSearch()

print("\nSystem components loaded successfully.")


# =========================================================
# ALL POSSIBLE INPUT VALUES
# =========================================================
activities = [
    "coffee_shop_hopping",
    "dinner",
    "gym_to_caffe",
    "hiking",
    "smart_casual"
]

undertones = [
    "warm",
    "cool",
    "neutral"
]

hijab_preferences = [
    "ya",
    "tidak"
]


# =========================================================
# GENERATE ALL COMBINATIONS
# 5 × 3 × 2 = 30 scenarios
# =========================================================
test_scenarios = list(
    itertools.product(
        activities,
        undertones,
        hijab_preferences
    )
)

print(
    f"\nTotal test scenarios: "
    f"{len(test_scenarios)}"
)


# =========================================================
# HELPER
# =========================================================
def normalize(value):
    return str(value).strip().lower()


# =========================================================
# STORAGE
# =========================================================
all_results = []
scenario_summaries = []

total_recommendations = 0
total_relevant = 0
successful_scenarios = 0
failed_scenarios = 0


# =========================================================
# RUN ALL 30 SCENARIOS
# =========================================================
for scenario_number, combination in enumerate(
    test_scenarios,
    start=1
):

    aktivitas, undertone, hijab = combination

    user_input = {
        "aktivitas": aktivitas,
        "undertone": undertone,
        "hijab": hijab
    }

    print("\n")
    print("=" * 80)
    print(
        f"SCENARIO "
        f"{scenario_number}/{len(test_scenarios)}"
    )
    print("=" * 80)

    print(f"Activity  : {aktivitas}")
    print(f"Undertone : {undertone}")
    print(f"Hijab     : {hijab}")


    # =====================================================
    # 1. RULE-BASED + METADATA FILTERING
    # =====================================================
    try:

        rule_result = rule_filter.filter(
            user_input
        )

    except Exception as e:

        print(
            f"\n[FAILED] Rule-Based Error: {e}"
        )

        failed_scenarios += 1

        scenario_summaries.append({
            "scenario": scenario_number,
            "aktivitas": aktivitas,
            "undertone": undertone,
            "hijab": hijab,
            "total_candidate": 0,
            "relevant_top3": 0,
            "relevance_at_3": 0,
            "status": "FAILED"
        })

        continue


    recommended_colors = [
        normalize(color)
        for color
        in rule_result["recommended_colors"]
    ]

    candidate_ids = rule_result[
        "candidate_ids"
    ]

    candidate_df = rule_result[
        "candidate_df"
    ]

    query = rule_result[
        "query"
    ]


    print("\nRecommended Colors:")

    for color in recommended_colors:

        print(
            "-",
            color.replace("_", " ").title()
        )


    total_candidate = rule_result[
        "total_candidate"
    ]

    print(
        f"\nTotal Candidate : "
        f"{total_candidate}"
    )


    # =====================================================
    # CHECK EMPTY CANDIDATE
    # =====================================================
    if total_candidate == 0:

        print(
            "\n[FAILED] "
            "No outfit candidate found."
        )

        failed_scenarios += 1

        scenario_summaries.append({
            "scenario": scenario_number,
            "aktivitas": aktivitas,
            "undertone": undertone,
            "hijab": hijab,
            "total_candidate": 0,
            "relevant_top3": 0,
            "relevance_at_3": 0,
            "status": "NO CANDIDATE"
        })

        continue


    # =====================================================
    # 2. TEXT ENCODING
    # =====================================================
    try:

        query_embedding = embedder.encode_text(
            query
        )

    except Exception as e:

        print(
            f"\n[FAILED] "
            f"FashionSigLIP Error: {e}"
        )

        failed_scenarios += 1
        continue


    # =====================================================
    # 3. FAISS SEARCH
    # =====================================================
    try:

        results = faiss_search.search(
            query_embedding,
            top_k=3,
            candidate_ids=candidate_ids
        )

    except Exception as e:

        print(
            f"\n[FAILED] "
            f"FAISS Error: {e}"
        )

        failed_scenarios += 1
        continue


    # =====================================================
    # 4. TOP-3 RELEVANCE VALIDATION
    # =====================================================
    print("\n" + "-" * 80)
    print("TOP-3 RELEVANCE VALIDATION")
    print("-" * 80)

    scenario_relevant = 0


    for rank, result in enumerate(
        results,
        start=1
    ):

        # =================================================
        # Cari metadata hasil berdasarkan ID
        # =================================================
        matched = candidate_df[
            candidate_df["id"].astype(int)
            == int(result["id"])
        ]


        if matched.empty:

            result_activity = "-"
            result_undertone = "-"
            result_hijab = "-"
            result_color = "-"

            activity_match = False
            undertone_match = False
            hijab_match = False
            color_match = False

            is_relevant = False


        else:

            row = matched.iloc[0]

            result_activity = normalize(
                row["aktivitas"]
            )

            result_undertone = normalize(
                row["undertone"]
            )

            result_hijab = normalize(
                row["hijab"]
            )

            result_color = normalize(
                row["warna_dominan"]
            )


            # =================================================
            # RELEVANCE CRITERIA
            # =================================================
            activity_match = (
                result_activity
                == normalize(aktivitas)
            )

            undertone_match = (
                result_undertone
                == normalize(undertone)
            )

            hijab_match = (
                result_hijab
                == normalize(hijab)
            )

            color_match = (
                result_color
                in recommended_colors
            )


            is_relevant = (
                activity_match
                and undertone_match
                and hijab_match
                and color_match
            )


        # =================================================
        # COUNT
        # =================================================
        if is_relevant:

            status = "RELEVANT"

            scenario_relevant += 1
            total_relevant += 1

        else:

            status = "NOT RELEVANT"


        total_recommendations += 1


        # =================================================
        # PRINT RESULT
        # =================================================
        print(
            f"{rank}. "
            f"{result['nama_gambar']} | "
            f"Activity: {result_activity} | "
            f"Undertone: {result_undertone} | "
            f"Hijab: {result_hijab} | "
            f"Color: {result_color} | "
            f"Similarity: "
            f"{result['score']:.4f} | "
            f"{status}"
        )


        # =================================================
        # SAVE DETAIL
        # =================================================
        all_results.append({

            "scenario": scenario_number,

            "input_aktivitas":
                aktivitas,

            "input_undertone":
                undertone,

            "input_hijab":
                hijab,

            "recommended_colors":
                ", ".join(
                    recommended_colors
                ),

            "total_candidate":
                total_candidate,

            "rank":
                rank,

            "id":
                result["id"],

            "nama_gambar":
                result["nama_gambar"],

            "result_aktivitas":
                result_activity,

            "result_undertone":
                result_undertone,

            "result_hijab":
                result_hijab,

            "warna_dominan":
                result_color,

            "similarity":
                result["score"],

            "activity_match":
                activity_match,

            "undertone_match":
                undertone_match,

            "hijab_match":
                hijab_match,

            "color_match":
                color_match,

            "relevant":
                is_relevant
        })


    # =====================================================
    # RELEVANCE@3
    # =====================================================
    relevance_at_3 = (
        scenario_relevant / 3
    ) * 100


    print("\n" + "-" * 80)

    print(
        f"Relevant Recommendations : "
        f"{scenario_relevant}/3"
    )

    print(
        f"Relevance@3              : "
        f"{relevance_at_3:.2f}%"
    )


    # =====================================================
    # SCENARIO STATUS
    # =====================================================
    if len(results) == 3:

        successful_scenarios += 1
        scenario_status = "SUCCESS"

    else:

        failed_scenarios += 1
        scenario_status = "INCOMPLETE"


    scenario_summaries.append({

        "scenario":
            scenario_number,

        "aktivitas":
            aktivitas,

        "undertone":
            undertone,

        "hijab":
            hijab,

        "total_candidate":
            total_candidate,

        "relevant_top3":
            scenario_relevant,

        "relevance_at_3":
            relevance_at_3,

        "status":
            scenario_status
    })


# =========================================================
# OVERALL RELEVANCE
# =========================================================
if total_recommendations > 0:

    overall_relevance = (
        total_relevant
        / total_recommendations
    ) * 100

else:

    overall_relevance = 0


# =========================================================
# FINAL SUMMARY
# =========================================================
print("\n\n")
print("=" * 80)
print("FINAL RELEVANCE@3 TEST RESULT")
print("=" * 80)

print(
    f"Total Input Combinations     : "
    f"{len(test_scenarios)}"
)

print(
    f"Successful Scenarios         : "
    f"{successful_scenarios}"
)

print(
    f"Failed/Incomplete Scenarios  : "
    f"{failed_scenarios}"
)

print(
    f"Total Recommendations Tested : "
    f"{total_recommendations}"
)

print(
    f"Relevant Recommendations     : "
    f"{total_relevant}"
)

print(
    f"Not Relevant Recommendations : "
    f"{total_recommendations - total_relevant}"
)

print(
    f"Overall Relevance@3          : "
    f"{overall_relevance:.2f}%"
)

print("=" * 80)


# =========================================================
# PRINT SCENARIO SUMMARY
# =========================================================
print("\n")
print("=" * 80)
print("SUMMARY PER SCENARIO")
print("=" * 80)

for summary in scenario_summaries:

    print(
        f"{summary['scenario']:02d}. "
        f"{summary['aktivitas']:<22} | "
        f"{summary['undertone']:<7} | "
        f"{summary['hijab']:<5} | "
        f"Candidates: "
        f"{summary['total_candidate']:<4} | "
        f"Relevant: "
        f"{summary['relevant_top3']}/3 | "
        f"R@3: "
        f"{summary['relevance_at_3']:.2f}% | "
        f"{summary['status']}"
    )


# =========================================================
# SAVE RESULTS TO CSV
# =========================================================
os.makedirs(
    "evaluation",
    exist_ok=True
)


# Detail 90 recommendations
detail_df = pd.DataFrame(
    all_results
)

detail_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# Summary 30 scenarios
summary_df = pd.DataFrame(
    scenario_summaries
)

summary_df.to_csv(
    "evaluation/relevance_scenario_summary.csv",
    index=False
)


print("\nResults saved to:")

print(
    "- evaluation/"
    "relevance_test_results.csv"
)

print(
    "- evaluation/"
    "relevance_scenario_summary.csv"
)

print("\nTesting completed.")