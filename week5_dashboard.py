import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# Cấu hình trang web
st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="centered")

# --- 1. HÀM TẢI VÀ HUẤN LUYỆN MÔ HÌNH (Kết hợp Tuần 1-4) ---
@st.cache_resource # Giúp lưu bộ nhớ đệm, không phải train lại mỗi lần bấm nút
def train_model():
    # Đọc dữ liệu sạch
    df = pd.read_csv('cleaned_house_prices.csv')
    
    # Lấy các đặc trưng tốt nhất
    features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'Distance_to_Center_km']
    X = df[features]
    y = df['SalePrice']
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Huấn luyện mô hình XGBoost (Quán quân của Tuần 3-4)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

# Tải mô hình
model, scaler = train_model()

# --- 2. XÂY DỰNG GIAO DIỆN WEB (Tuần 5) ---
st.title("🏡 AI House Price Predictor")
st.markdown("Dự án ADY201m - Tích hợp Đặc trưng Không gian (Geospatial) bằng **XGBoost**")
st.divider()

st.sidebar.header("⚙️ Nhập thông số căn nhà")

# Tạo các thanh trượt và ô nhập liệu cho người dùng
overall_qual = st.sidebar.slider("Điểm chất lượng tổng thể (1-10)", 1, 10, 6)
gr_liv_area = st.sidebar.number_input("Diện tích sống (Square Feet)", min_value=500, max_value=5000, value=1500)
garage_cars = st.sidebar.slider("Sức chứa Garage (Số ô tô)", 0, 4, 2)
total_bsmt_sf = st.sidebar.number_input("Diện tích tầng hầm (Square Feet)", min_value=0, max_value=3000, value=1000)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Vị trí địa lý")
distance_km = st.sidebar.slider("Khoảng cách đến Trung tâm (km)", 0.0, 20.0, 5.0, step=0.1)

# --- 3. XỬ LÝ DỰ ĐOÁN ---
st.subheader("📊 Kết quả định giá")

# Đóng gói dữ liệu người dùng nhập thành DataFrame
user_data = pd.DataFrame({
    'OverallQual': [overall_qual],
    'GrLivArea': [gr_liv_area],
    'GarageCars': [garage_cars],
    'TotalBsmtSF': [total_bsmt_sf],
    'Distance_to_Center_km': [distance_km]
})

if st.button("Dự đoán Giá Nhà", type="primary", use_container_width=True):
    with st.spinner('AI đang phân tích dữ liệu...'):
        # Chuẩn hóa dữ liệu đầu vào giống hệt lúc huấn luyện
        user_data_scaled = scaler.transform(user_data)
        
        # Dự đoán
        prediction = model.predict(user_data_scaled)[0]
        
        # Hiển thị kết quả
        st.success("Hoàn tất!")
        st.metric(label="Mức giá đề xuất", value=f"${prediction:,.2f} USD")
        
        # Thêm phần giải thích
        st.info(f"💡 **AI Insight:** Căn nhà cách trung tâm **{distance_km}km** với chất lượng **{overall_qual}/10** có giá trị ước tính khoảng **${prediction:,.0f}**. Mô hình XGBoost đã điều chỉnh mức giá này dựa trên các quy luật phi tuyến tính học được từ dữ liệu lịch sử.")