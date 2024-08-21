import json
import random

# JSON 파일 로드 함수
json_file = "llm/matjib.json"

def load_restaurants(file_name):
    """JSON 파일을 로드하여 맛집 데이터를 반환"""
    with open(file_name, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data["맛집"]

# 맛집 추천 함수
def restaurant_bot(lunch: bool, dinner: bool, alcohol: bool, cafe: bool) -> str:
    restaurants = load_restaurants(json_file)
    max_attempts = 20
    if lunch:
        for _ in range(max_attempts):
            restaurant = random.choice(restaurants)
            if restaurant["점심밥"]:
                return f"추천 맛집: {restaurant['이름']}, 특징: {', '.join(restaurant['특징'])}"
    elif dinner:
        for _ in range(max_attempts):
            restaurant = random.choice(restaurants)
            if restaurant["저녁밥"]:
                return f"추천 맛집: {restaurant['이름']}, 특징: {', '.join(restaurant['특징'])}"
    elif alcohol:
        for _ in range(max_attempts):
            restaurant = random.choice(restaurants)
            if restaurant["술"]:
                return f"추천 맛집: {restaurant['이름']}, 특징: {', '.join(restaurant['특징'])}"
    elif cafe:
        for _ in range(max_attempts):
            restaurant = random.choice(restaurants)
            if restaurant["카페"]:
                return f"추천 맛집: {restaurant['이름']}, 특징: {', '.join(restaurant['특징'])}"
    else:
        return "죄송합니다. 조건에 맞는 맛집을 찾지 못했습니다."
    
                
                
                
if __name__ == "__main__":
    # 함수 사용 예시
    lunch = True
    dinner = True
    alcohol = False
    cafe = False
    
    result = restaurant_bot(lunch, dinner, alcohol, cafe)
    print(result)