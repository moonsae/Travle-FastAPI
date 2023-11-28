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
import math
import time


app=FastAPI()

origins=["http://localhost:8091/prj/recTour",
         "http://localhost:8091",
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


@app.post("/recommend")
async def recommendSpot(request: Request):
    start=time.time()
    data_list  = await request.json()
    content_ids = []
    overviews_list = []
    for data in data_list:
        contentid = data.get("contentid")
        overview = data.get("overview")
        if overview:
            content_ids.append(contentid)
            overviews_list.append(overview)
    #장소에 대한 데이터와 명사로만 분리한 데이터
    spotList = load_data()
    spot=spotList['DATA']
    
    foodList=load_food_data()
    restaurant=foodList['DATA']
    
    #모델 로드
    model = load_model()
    
    #클라이언트가 선택한 장소 데이터 가공
    selectSpot = select_spot(overviews_list)
    
    #장소데이터의 d2v
    spotVec = get_document_vectors(spot,model)
    #식당데이터벡터
    restaurantVec = get_document_vectors(restaurant,model)
    #선택데이터벡터
    myVec = get_document_vectors(selectSpot,model)
    
    #관광지와 식당의 코사인유사도 계산
    spot_consin_sim = cosine_similarity(spotVec,myVec)
    restaurant_consin_sim = cosine_similarity(restaurantVec,myVec)
    
    #추천 결과 추출
    spot_recommend_list=recommend_result(spotList,spot_consin_sim)
    restaurant__recommend_list=recommend_result(foodList,restaurant_consin_sim)
    
    spot_recommend_title = spot_recommend_list[0]
    spot_response_message =[{"index":index,"contentid": int(id)} for index, id in spot_recommend_title.items()]
    spot_recommed_per=spot_recommend_list[1]
    
    restaurant_recommend_title = restaurant__recommend_list[0]
    restaurant_response_message =[{"index":index+5,"contentid": int(id)} for index, id in restaurant_recommend_title.items()]
    restaurant_recommed_per=restaurant__recommend_list[1]
    print(spot_recommed_per)
    print(spot_response_message)
    print("--------------------")
    print(restaurant_recommed_per)
    print(restaurant_response_message)
    
    response_message = spot_response_message+restaurant_response_message
    end=time.time()
    print(end-start)
    return response_message

