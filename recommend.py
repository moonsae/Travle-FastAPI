import pandas as pd
from konlpy.tag import Okt
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from stopword import *

okt=Okt()

# 이 함수는 사용자가 입력한 장소의 데이터를 가공해야 하는 것 
def select_spot(likeList):
    #print(likeList)
    stop_words = seoul_stop_words()
    select_spot_word=[]
    row_nouns = [] 
    for i in range(len(likeList)):
        for word in okt.nouns(likeList[i]):
            if word not in stop_words:
                row_nouns.append(word)
    select_spot_word.append(row_nouns)
    return select_spot_word
# 모델과 csv 파일을 read 해야 함
def load_data():
    spotList=pd.read_csv('data\seoul_place.csv')
    return spotList

def load_food_data():
    foodList=pd.read_csv('data\seoul_food.csv')
    return foodList

def load_model():
    model_filename='seoulw2v.model'
    model = Word2Vec.load('C:/Users/user/Desktop/데이터파일/서울특별시/'+model_filename)
    return model
def get_document_vectors(document_list,word2vec_model):
    document_embedding_list = []
    # 각 문서에 대해서
    for line in range(len(document_list)):
        doc2vec = None
        count = 0
        for word in document_list[line]:
            if word in word2vec_model.wv.key_to_index:
                count += 1
                # 해당 문서에 있는 모든 단어들의 벡터값을 더한다.
                if doc2vec is None:
                    doc2vec = word2vec_model.wv[word]
                else:
                    doc2vec = doc2vec + word2vec_model.wv[word]

        if doc2vec is not None:
            # 단어 벡터를 모두 더한 벡터의 값을 문서 길이로 나눠준다.
            doc2vec = doc2vec / count
            document_embedding_list.append(doc2vec)
    # 각 문서에 대한 문서 벡터 리스트를 리턴
    return document_embedding_list

def recommend_result(dataList,cosine_sim):
    #spot_List=dataList[['CONTENTID','FIRSTIMAGE']]
    rec_List=dataList[['CONTENTID']]
    indices = pd.Series(dataList.index, index = dataList['CONTENTID']).drop_duplicates()
    
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]

    spot_indices = [i[0] for i in sim_scores]

    # 전체 데이터프레임에서 해당 인덱스의 행만 추출. 5개의 행을 가진다.
    recommend = dataList.iloc[spot_indices].reset_index(drop=True)
    recommendTitle = recommend['CONTENTID']
    
    return recommendTitle,sim_scores