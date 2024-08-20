import json
import random

# JSON 파일 로드 함수
json_file = "/home/elicer/LLM_project_YAI/llm/matjib.json"

def load_restaurants(file_name):
    """JSON 파일을 로드하여 맛집 데이터를 반환"""
    with open(file_name, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data["맛집"]

# 맛집 추천 함수
def restaurant_bot(lunch: bool, dinner: bool, alcohol: bool, cafe: bool) -> str:
    restaurants = load_restaurants(json_file)
    max_attempts = 20

    for _ in range(max_attempts):
            print("머먹지 함수 실행됨")
            restaurant = random.choice(restaurants)
            if (restaurant["점심밥"] == lunch or 
                restaurant["저녁밥"] == dinner or 
                restaurant["술"] == alcohol or 
                restaurant["카페"] == cafe):
                return f"추천 맛집: {restaurant['이름']}, 특징: {', '.join(restaurant['특징'])}"
if __name__ == "__main__":
    # 함수 사용 예시
    lunch = True
    dinner = True
    alcohol = False
    cafe = False
    
    result = restaurant_bot(lunch, dinner, alcohol, cafe)
    print(result)