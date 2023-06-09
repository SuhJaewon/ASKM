import cv2
import pytesseract
import numpy as np
from PIL import Image
import sys
import streamlit as st
import re
import csv
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

def conv_pil_to_cv(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
def preprocess_image(image: Image) -> Image:
    upscale_ratio = 7
    resized_image = image.resize((image.width * upscale_ratio, image.height * upscale_ratio))
    enhanced_image = cv2.detailEnhance(conv_pil_to_cv(resized_image), sigma_s=60, sigma_r=10)
    return Image.fromarray(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
def receipt_analysis():
    st.title("영수증 분류기")
    st.write("영수증 이미지를 업로드하여 수량과 가격을 분류합니다.")
    uploaded_image = st.file_uploader("영수증 이미지 업로드", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        preprocessed_image = preprocess_image(image)
        image_cv = conv_pil_to_cv(preprocessed_image)
        image_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(image_gray, ksize=(11, 11), sigmaX=0)
        ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
        edged = cv2.Canny(blur, 10, 250)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x_min, x_max = sys.maxsize, -sys.maxsize
        y_min, y_max = sys.maxsize, -sys.maxsize
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            x_max = max(x_max, x + w)
            y_min = min(y_min, y)
            y_max = max(y_max, y + h)
        trimmed_image = image_cv[y_min:y_max, x_min:x_max]
        trimmed_image_pil = Image.fromarray(cv2.cvtColor(trimmed_image, cv2.COLOR_BGR2RGB))
        st.image(trimmed_image_pil, caption="자른 이미지", use_column_width=True)
        my_config = "-l new+new1 --oem 1 --psm 4 -c preserve_interword_spaces=1"
        extracted_text = pytesseract.image_to_string(trimmed_image_pil, config=my_config)
        return extracted_text
def extract_product_info(text):
    item_pattern = r'([\w\s]+)\s+(\d+\.\d+)\s+([\d,]+)'
    product_info = []
    matches = re.findall(item_pattern, text)
    for match in matches:
        product_name = match[0].replace('ITEM NAME', '').replace('ary','').replace('0TY', '').replace('AMOUNT', '').replace('QTY','').strip()
        quantity = float(match[1])
        price = int(match[2].replace(',', ''))
        product_info.append((product_name, quantity, price))
    return product_info

def income_expense_management(ocr_text):
    productinfos = extract_product_info(ocr_text)
    totalapp = st.text_input('예산을 입력해주세요', value='0')

    totalspn = 0
    for row in productinfos:
        totalspn += int(row[2])
        if totalspn > int(totalapp):
            st.write('예산 초과!')
            break
    moneyleft = int(totalapp) - int(totalspn)
    foodspend = 0
    otherspend = 0
    for row in productinfos:
        if 'food' in row[0]:  # '식품' 키워드를 기준으로 카테고리 분류
            foodspend += int(row[2])
        else:
            otherspend += int(row[2])
    left_used_categories = [moneyleft, foodspend, otherspend]
    for i in range(len(left_used_categories)):
        if left_used_categories[i] < 0:
            left_used_categories[i] = 0
    labels = ['Money', 'Food spend', 'Other spend']
    # Remove labels with 0 values and update indices
    indices = [i for i, value in enumerate(left_used_categories) if value != 0]
    labels = [labels[i] for i in indices]
    # Create a new list of values based on indices
    left_used_categories = [left_used_categories[i] for i in indices] 
    fig, ax = plt.subplots()
    w = {"edgecolor": "black", "linewidth": 3}
    ax.pie(left_used_categories, startangle=90, autopct='%.1f', wedgeprops=w)
    ax.legend(labels)
    st.pyplot(fig)
    total_budget = sum(int(row[2]) for row in productinfos)  # 예산 합계 계산        
    st.write("총지출: ", total_budget)  # 예산 합계 출력

    with open('dataset.csv', encoding='utf8') as f:
        data = csv.reader(f)
        next(data)
        data = list(data)
        
        for product in productinfos:
            product_name, quantity, price = product
            st.write(f'상품명: {product_name}')
            st.write(f'수량: {quantity}')
            st.write(f'가격: {price} 동')
            st.write('---')

    return productinfos

def budget_management(product_info_list):
    # 예산 관리 기능을 구현해야 합니다.
    pass
def main():
    st.title("ASKM")
    menu = st.sidebar.selectbox("메뉴", ["영수증 자동 분석"])
    if menu == "영수증 자동 분석":
        ocr_text = receipt_analysis()
        if ocr_text:
            product_info_list = income_expense_management(ocr_text)
            budget_management(product_info_list)
        budget_management([])
if __name__ == "__main__":
    main()
