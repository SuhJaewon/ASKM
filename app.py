import cv2
import pytesseract
import numpy as np
from PIL import Image
import sys
import streamlit as st
import re
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

# 영수증 자동 분석
def receipt_analysis():
    st.title("영수증 분류기")
    st.write("영수증 이미지를 업로드하여 수량과 가격을 분류합니다.")

    uploaded_image = st.file_uploader("영수증 이미지 업로드", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        # 이미지 전처리
        preprocessed_image = preprocess_image(image)

        # OpenCV 배열로 변환
        image_cv = conv_pil_to_cv(preprocessed_image)

        # 이미지 처리
        image_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image_gray, ksize=(9, 9), sigmaX=0)
        ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
        edged = cv2.Canny(blur, 10, 250)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 컨투어 박스 추출
        x_min, x_max = sys.maxsize, -sys.maxsize
        y_min, y_max = sys.maxsize, -sys.maxsize

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            x_max = max(x_max, x + w)
            y_min = min(y_min, y)
            y_max = max(y_max, y + h)

        # 이미지 자르기
        trimmed_image = image_cv[y_min:y_max, x_min:x_max]
        trimmed_image_pil = Image.fromarray(cv2.cvtColor(trimmed_image, cv2.COLOR_BGR2RGB))

        # 잘린 이미지 출력
        st.image(trimmed_image_pil, caption="자른 이미지", use_column_width=True)

        my_config = "-l new+new1 --oem 1 --psm 4 -c preserve_interword_spaces=1"

        # OCR 적용
        extracted_text = pytesseract.image_to_string(trimmed_image_pil, config=my_config)  # 영어와 베트남어 언어 모델 설정

        # 추출된 텍스트 반환
        return extracted_text

def extract_product_info(text):
    # 텍스트에서 상품명과 가격을 추출하기 위한 정규 표현식
    item_pattern = r'([\w\s]+)\s+(\d+\.\d+)\s+([\d,]+)'

    product_info = []

    # 텍스트에서 상품명과 가격 추출
    matches = re.findall(item_pattern, text)
    for match in matches:
        product_name = match[0].replace('ITEM NAME', '').replace('0TY', '').replace('AMOUNT', '').replace('QTY','').strip()
        quantity = float(match[1])
        price = match[2].replace(',', '')
        product_info.append((product_name, quantity, price))

    return product_info

# 기능 2: 수입과 지출 관리
def income_expense_management(ocr_text):
    product_info = extract_product_info(ocr_text)
    for product in product_info:
        product_name, quantity, price = product
        st.write(f'상품명: {product_name}')
        st.write(f'수량: {quantity}')
        st.write(f'가격: {price} 동')
        st.write('---')

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
        ocr_text = receipt_analysis()
        if ocr_text:
            income_expense_management(ocr_text)

    elif menu == "수입과 지출 관리":
        st.write("수입과 지출 관리 기능을 선택하셨습니다.")

    elif menu == "예산 관리":
        budget_management()

# 웹앱 실행
if __name__ == "__main__":
    main()
