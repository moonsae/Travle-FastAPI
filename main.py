import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.requests import Request
from pydantic import BaseModel
from recommend import *
from sklearn.metrics.pairwise import cosine_similarity



app=FastAPI()

origins=["http://129.154.56.13:8091/tour/recTour",
         "http://129.154.56.13:8091",
         ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return{"message":"home"} 

@app.post("/api/recommend")
async def recommendSpot(request: Request):
    data = await request.json()
    
    # JSON 문자열을 파이썬 객체로 변환
    data = jsonable_encoder(data)
    
    selected_area = data.get("selectedArea")
    select_places = data.get("selectPlaces")

    content_title = []
    content_ids = []
    overviews_list = []
    
    for data_real in select_places:
        title=data_real.get("title")
        contentid = data_real.get("contentid")
        overview = data_real.get("overview")
        if overview:
            content_title.append(title)
            content_ids.append(contentid)
            overviews_list.append(overview)

    print(selected_area)
    print(content_ids)
    print(content_title)
    #print(overviews_list)
    
    #장소에 대한 데이터와 명사로만 분리한 데이터
    spotList = load_spot_data(selected_area)
    spot=spotList['DATA']
    
    foodList= load_food_data(selected_area)
    restaurant=foodList['DATA']
    
    #모델 로드
    model = Word2Vec.load('models/recw2v.model')
    
    #클라이언트가 선택한 장소 데이터 가공
    selectSpot = select_spot(overviews_list,content_title)
    
    # 벡터 계산
    spotVec = get_document_vectors(spot, model)
    restaurantVec = get_document_vectors(restaurant, model)
    myVec = get_document_vectors(selectSpot, model)

    # numpy 배열로 변환
    spotVec_2d = np.array(spotVec)
    restaurantVec_2d = np.array(restaurantVec)
    myVec_2d = np.array(myVec)

    #관광지와 식당의 코사인 유사도 계산
    spot_consine_sim = cosine_similarity(spotVec_2d, myVec_2d)
    restaurant_consine_sim = cosine_similarity(restaurantVec_2d, myVec_2d)

    
    #추천 결과 추출
    spot_recommend_list=recommend_result(spotList,spot_consine_sim)
    restaurant__recommend_list=recommend_result(foodList,restaurant_consine_sim)
    
    spot_recommend_title = spot_recommend_list[0]
    spot_response_message =[{"index":index,"contentid": int(id)} for index, id in spot_recommend_title.items()]
    spot_recommed_per=spot_recommend_list[1]
    
    restaurant_recommend_title = restaurant__recommend_list[0]
    restaurant_response_message =[{"index":index+6,"contentid": int(id)} for index, id in restaurant_recommend_title.items()]
    restaurant_recommed_per=restaurant__recommend_list[1]
    print(spot_recommed_per)
    print(spot_response_message)
    print("-----------------------------------------------")
    print(restaurant_recommed_per)
    print(restaurant_response_message)
    
    response_message = spot_response_message+restaurant_response_message
    return response_message

    