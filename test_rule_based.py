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


# ==================================================
# Load Rule-Based System
# ==================================================
rule_based = RuleBasedFilter(
    "metadata/dataset_outfit_final_revisi.csv"
)


# ==================================================
# Contoh Input
# ==================================================
user_input = {
    "aktivitas": "dinner",
    "undertone": "cool",
    "hijab": "ya"
}


# ==================================================
# Jalankan Rule-Based
# ==================================================
result = rule_based.filter(user_input)


# ==================================================
# Tampilkan Hasil
# ==================================================
print("\n========================================")
print("USER INPUT")
print("========================================")

print(
    "Aktivitas :",
    user_input["aktivitas"]
)

print(
    "Undertone :",
    user_input["undertone"]
)

print(
    "Hijab     :",
    user_input["hijab"]
)


print("\n========================================")
print("RULE-BASED RESULT")
print("========================================")

print("Recommended Colors:")

for color in result["recommended_colors"]:

    print(
        "-",
        color.replace("_", " ").title()
    )


print("\nTotal Candidate :",
      result["total_candidate"])


print("\n========================================")
print("FASHIONSIGLIP QUERY")
print("========================================")

print(result["query"])