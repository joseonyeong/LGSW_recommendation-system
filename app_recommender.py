import numpy as np
import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import surprise

st.title("영화 추천 시스템")
# st.write(f"Surprise 버전 : {surprise.__version__}")

movies = pd.read_csv('data/ml-latest-small/movies.csv')
ratings = pd.read_csv('data/ml-latest-small/ratings.csv')
links = pd.read_csv('data/ml-latest-small/links.csv')
images = pd.read_csv('data/ml-latest-small/images.csv').rename(
    columns={'item_id' : 'movieId', 'image':'image_url'}
)

st.write(movies.head())

def train_recommendation_model():
        """
            프로세스:
            1. Reader 클래스 활용 사용자의 평점 범위 설정
            2. ㅇㅁㅅㅁㄴㄷㅅ rorcp 
        """
    return None