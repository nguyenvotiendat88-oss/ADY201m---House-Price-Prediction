import streamlit as st
import pandas as pd
import joblib

# 1. Cấu hình trang (Phải đặt ở dòng đầu tiên)
st.set_page_config(page_title="House Price Prediction", page_icon="🏡", layout="centered")

# 2. Tiêu đề và Header
st.title("🏡 House Price Prediction App")
st.markdown("**ADY201m Course Project**")
st.divider() # Đường kẻ ngang phân cách

# 3. Load mô hình (Dùng cache để tải nhanh hơn)
@st.cache_resource 
def load_model():
    mo_hinh = joblib.load('best_xgboost_model.pkl')
    cai_can = joblib.load('scaler.pkl')
    return mo_hinh, cai_can

try:
    mo_hinh, cai_can = load_model()
except:
    st.error("⚠️ Lỗi: Không tìm thấy file mô hình. Vui lòng kiểm tra lại!")
    st.stop()

# 4. Thanh công cụ bên trái (Sidebar)
st.sidebar.header("⚙️ Thông Số Căn Nhà")
st.sidebar.markdown("Điều chỉnh các thông số dưới đây:")

diem_nha = st.sidebar.slider("🌟 Điểm chất lượng (1-10)", 1, 10, 5)
dien_tich = st.sidebar.number_input("📐 Diện tích sống (sqft)", min_value=500, max_value=5000, value=1500, step=50)
xe_oto = st.sidebar.number_input("🚗 Sức chứa Garage (số xe)", min_value=0, max_value=4, value=2)
tang_ham = st.sidebar.number_input("🕳️ Diện tích tầng hầm (sqft)", min_value=0, max_value=3000, value=1000, step=50)
km = st.sidebar.slider("📍 Khoảng cách đến trung tâm (km)", 0.0, 20.0, 5.0, step=0.5)

# 5. Khu vực hiển thị kết quả chính
st.markdown("### 📊 Dự báo Giá Trị Bất Động Sản")
st.markdown("Hệ thống sử dụng thuật toán **XGBoost** để phân tích dữ liệu từ khu vực Ames, Iowa.")

if st.button("🚀 Phân Tích & Định Giá", use_container_width=True):
    with st.spinner("AI đang xử lý dữ liệu..."):
        # Tiền xử lý
        data_vo = pd.DataFrame({
            'OverallQual': [diem_nha],
            'GrLivArea': [dien_tich],
            'GarageCars': [xe_oto],
            'TotalBsmtSF': [tang_ham],
            'Distance_to_Center_km': [km]
        })
        
        data_scale = cai_can.transform(data_vo)
        gia_do_la = mo_hinh.predict(data_scale)[0]
        
        # Dùng tỷ giá 26,343 cho độ chính xác cao
        gia_tien_viet = (gia_do_la * 26343) / 1_000_000_000 
        
        st.success("✅ Phân tích hoàn tất!")
        
        # Hiển thị số liệu bằng thẻ Metric
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="💵 Mức giá ước tính (USD)", value=f"${gia_do_la:,.0f}")
        with col2:
            st.metric(label="🇻🇳 Quy đổi", value=f"{gia_tien_viet:.2f} Tỷ VNĐ")
        
        st.divider()
        st.info(f"💡 **AI Insight:** Căn nhà cách trung tâm **{km}km**, diện tích **{dien_tich} sqft** với mức độ hoàn thiện **{diem_nha}/10** được hệ thống định giá khoảng **{gia_tien_viet:.2f} Tỷ VNĐ**. Mô hình XGBoost này hiện đạt độ tin cậy (R2 Score) là 88%.")
else:
    st.info("👈 Hãy điều chỉnh thông số bên thanh công cụ và bấm nút Phân tích để xem kết quả.")