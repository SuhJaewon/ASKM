import streamlit as st
import pytesseract
import re
import joblib
import sqlite3
import cv2
import numpy as np
from PIL import Image


# 기능 1: 영수증 자동 분석 기능
def receipt_analysis():
    # 사용자 인터페이스 구현
    st.title('영수증 분석기')


    # 사진 촬영 및 처리 코드 작성
    st.write("사진을 촬영해주세요.")

    # 사진 촬영 버튼 클릭 시
    if st.button('사진 촬영'):
        # 카메라 초기화 및 사진 촬영
        camera = cv2.VideoCapture(0)
        _, frame = camera.read()

        # 이미지 저장
        cv2.imwrite('captured_image.jpg', frame)

        # 이미지 크기 조정
        image = Image.open('captured_image.jpg')
        resized_image = image.resize((800, 600))
        resized_image.save('resized_image.jpg')

        # 이미지 이진화
        gray_image = cv2.cvtColor(np.array(resized_image), cv2.COLOR_BGR2GRAY)
        threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cv2.imwrite('threshold_image.jpg', threshold_image)

        # OCR을 이용한 텍스트 추출
        text = pytesseract.image_to_string(threshold_image, lang='kor')

        # 처리된 결과 표시
        st.write("OCR 처리된 텍스트 결과:")
        st.write(text)

    

    




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
