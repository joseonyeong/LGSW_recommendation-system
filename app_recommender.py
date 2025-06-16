import numpy as np 
import streamlit as st 
import pandas as pd 
from surprise import Dataset, Reader, SVD # 추천 알고리즘
from surprise.model_selection import train_test_split
import surprise

#  버전 표시 
st.title("영화 추천 시스템")
# st.write(f"Surprise 버전 : {surprise.__version__}")

movies = pd.read_csv("data/ml-latest-small/movies.csv")
ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
links = pd.read_csv("data/ml-latest-small/links.csv")
images = pd.read_csv("data/ml-latest-small/ml1m_images.csv").rename(
    columns={'item_id' : 'movieId', 'image' : 'image_url'}
)

# st.write(movies.head())
# st.write(ratings.head())
# st.write(links.head())
# st.write(images.head())

def train_recommendation_model():
    """
    프로세스 : 
    1. Reader 클래스 활용 사용자의 평점 범위 설정
    2. Dataset 객체 생성 및 데이터 가져오기
    3. 학습 데이터 분할 (80%) (20%)
    4. SVD 모델 학습 수행
    """
    # 평점의 범위를 0.5~5.0으로 지정하여 Reader 객체 생성
    reader = Reader(rating_scale=(0.5, 5.0))
    # Surprise용 데이터셋으로 변환 (userId, movieId, rating 열을 사용)
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    # 데이터셋을 학습용(80%)과 테스트용(20%)으로 분할
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    # SVD(Singular Value Decomposition) 알고리즘 모델 객체 생성
    model = SVD()
    # 학습 데이터를 기반으로 모델 학습 수행
    model.fit(trainset)
    # 학습된 모델 반환
    return model 

# 추천 모델 학습 함수 호출하여 모델 생성
model = train_recommendation_model()
# Streamlit에 모델 정보 출력
st.write(model)
# 학습 완료 메시지 표시
st.success("모델 학습 완료")


# 특정 사용자에게 영화를 추천하는 함수 정의
def get_top_n_recommend(user_id, model, movies_df, n=3):
    """
    특정 사용자에게 영화를 추천함
    Args:
        user_id (int) : 추천 받을 사용자의 ID
        model (SVD) : 학습된 SVD 모델 
        movies_df (pd.DataFrame) : 영화 정보가 담긴 데이터프레임
        n (int) : 추천할 영화의 개수 (기본값: 3)
    
    Returns:
        top_n_movies (pd.DataFrame) : 추천 영화 정보와 예측 평점이 포함된 데이터프레임
    """

    # 전체 영화 ID 목록을 가져옴
    movie_ids = movies_df['movieId'].unique() 
    # 해당 사용자가 이미 본 영화들의 평점 정보를 추출
    watched_movies = ratings[ratings['userId'] == user_id]

    # 사용자가 시청하지 않은 영화들에 대해서만 평점을 예측
    predictions = [
        model.predict(user_id, movie_id) for movie_id in movie_ids 
        if movie_id not in watched_movies['movieId'].values
    ]
    # 예측 평점이 높은 순으로 정렬 후 상위 n개만 추출
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

    # 추천된 영화들의 ID와 예측 평점을 데이터프레임으로 생성
    top_n_movies = pd.DataFrame({
        'movieId': [pred.iid for pred in top_n],             # 영화 ID
        'predicted_rating': [pred.est for pred in top_n]     # 예측 평점
    })

    # 추천 영화 정보에 제목 등의 상세 정보를 병합
    top_n_movies = top_n_movies.merge(movies_df, on='movieId', how='inner')

    # 최종 추천 결과 반환
    return top_n_movies

# Streamlit 버튼 생성: 사용자가 클릭하면 영화 추천이 실행됨
if st.button("추천 영화 보기"):
    # 추천 결과 표시를 위한 소제목 출력
    st.subheader("영화 추천 받기")

    # 사용자에게 user ID를 입력받는 숫자 입력 창 생성 (1~100 범위)
    user_id = st.number_input(
        "사용자 ID 입력하세요 (1-100)", 
        min_value=1,         # 최소값 1
        max_value=100,       # 최대값 100
        step=1               # 증가 단위는 1
    )

    # 추천 영화 리스트를 가져옴 (예: 상위 10개 추천)
    top_movies = get_top_n_recommend(user_id, model, movies, n=10)
    
    # 추천받은 각 영화에 대해 정보를 순차적으로 표시
    for idx, row in top_movies.iterrows():
        # 해당 영화의 이미지 URL 정보가 있는 행을 필터링
        movie_image_row = images[images['movieId'] == row['movieId']]
        
        # 이미지가 존재할 경우
        if not movie_image_row.empty:
            # 첫 번째 결과에서 이미지 URL 추출
            movie_image = movie_image_row['image_url'].values[0]
        else:
            # 이미지가 없을 경우 경고 메시지 출력 및 기본 이미지 사용
            st.warning(f"영화 ID {row['movieId']}의 이미지를 찾을 수 없습니다. 기본 이미지를 사용합니다.")
            movie_image = "https://via.placeholder.com/150"
        
        # Streamlit의 두 컬럼 레이아웃을 이용해 좌우로 영화 정보 표시
        col1, col2 = st.columns(2)

        # 왼쪽 열에 텍스트 정보 출력
        with col1:
            st.write(f"영화 ID: {row['movieId']}")       # 영화 ID 표시
            st.write(f"장르: {row['genres']}")           # 장르 정보 표시
        
        # 오른쪽 열에 이미지 출력
        with col2:
            st.image(
                movie_image,                             # 이미지 URL
                caption=f"{row['title']} (예상 평점: {row['predicted_rating']:.1f})",  # 이미지 하단 캡션
                use_container_width=True                 # 이미지가 열 너비에 맞게 표시됨
            )
