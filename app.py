import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Tesseract OCR 엔진 경로 설정
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

# PIL 이미지를 OpenCV의 numpy 배열로 변환하는 함수
def conv_pil_to_cv(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# 이미지 전처리 함수
def preprocess_image(image: Image) -> Image:
    # 이미지 업스케일링
    upscale_ratio = 7
    resized_image = image.resize((image.width * upscale_ratio, image.height * upscale_ratio))

    # 이미지 선명도 향상
    enhanced_image = cv2.detailEnhance(conv_pil_to_cv(resized_image), sigma_s=60, sigma_r=10)

    # OpenCV의 numpy 배열을 PIL 이미지로 변환하여 반환
    return Image.fromarray(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))


# 기능 1: 영수증 자동 분석 기능
def receipt_analysis():
    st.title("영수증 자동 분석")
    st.write("영수증 이미지를 업로드하여 자동 분석합니다.")

    uploaded_image = st.file_uploader("영수증 이미지 업로드", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        # 이미지 전처리
        preprocessed_image = preprocess_image(image)

        # 전처리된 이미지 출력
        st.image(preprocessed_image, caption="전처리된 이미지", use_column_width=True)

        my_config = "-l new+new1 --oem 1 --psm 6 -c preserve_interword_spaces=1"

        # OCR 적용
        extracted_text = pytesseract.image_to_string(preprocessed_image, config=my_config)  # 영어와 베트남어 언어 모델 설정

        # 추출된 텍스트 출력
        st.subheader("추출된 텍스트")
        st.text(extracted_text)
        print(extracted_text)


# 기능 2: 수입과 지출 관리
def income_expense_management():
    st.write("수입과 지출 관리 기능 구현 예시")


# 기능 3: 예산 관리
def budget_management():
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