from utils.recommendation import RecommendationEngine

engine = RecommendationEngine()

user = {
    "aktivitas": "dinner",
    "undertone": "warm",
    "hijab": "ya"
}

hasil = engine.recommend(user)

for h in hasil:
    print(h)