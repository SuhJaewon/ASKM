import streamlit as st

# 기능 1: 영수증 자동 분석 기능
def receipt_analysis():
    # 영수증 분석 코드 작성
    st.write("영수증 자동 분석 기능 구현 예시")

# 기능 2: 수입과 지출 관리
def income_expense_management():
    # 수입과 지출 관리 코드 작성
    st.write("수입과 지출 관리 기능 구현 예시")

# 기능 3: 예산 관리
def budget_management():
    # 예산 관리 코드 작성
    st.write("예산 관리 기능 구현 예시")

# 메인 페이지 레이아웃
def main():
    st.title("ASKM")

    # 사이드바 메뉴
    menu = st.sidebar.selectbox("메뉴", ["영수증 자동 분석", "수입과 지출 관리", "예산 관리"])

    # 메뉴에 따른 기능 호출
    if menu == "영수증 자동 분석":
        receipt_analysis()
    elif menu == "수입과 지출 관리":
        income_expense_management()
    elif menu == "예산 관리":
        budget_management()

# 웹앱 실행
if __name__ == "__main__":
    main()

import streamlit as st
import pytesseract
import re
from sklearn.externals import joblib
import sqlite3

# 데이터베이스 연결
conn = sqlite3.connect('receipts.db')
c = conn.cursor()

# OCR 설정
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 머신러닝 모델 로드
model = joblib.load('model.pkl')

# 사용자 인터페이스 구현
st.title('영수증 분석기')

uploaded_file = st.file_uploader('영수증 이미지 업로드', type=['jpg', 'png'])

if uploaded_file is not None:
    # 이미지 저장
    with open('receipt.jpg', 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # OCR을 이용한 텍스트 추출
    text = pytesseract.image_to_string('receipt.jpg')

    # 텍스트 전처리
    text = re.sub('[^0-9a-zA-Zㄱ-힗]', ' ', text)
    text = text.lower()

    # 머신러닝 모델을 이용한 카테고리 분류
    category = model.predict([text])[0]

    # 분류된 내용 데이터베이스에 저장
    c.execute("INSERT INTO receipts (text, category) VALUES (?, ?)", (text, category))
    conn.commit()

    # 분류된 내용 출력
    st.write('분류된 카테고리:', category)