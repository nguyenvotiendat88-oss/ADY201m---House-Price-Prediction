# 🏡 House Price Prediction with Geospatial Features

**Course:** ADY201m - AI & Data Science with Python & SQL
**University:** FPT University
**Dataset:** Ames Housing (Kaggle) supplemented with Geospatial Data

## 📌 Project Overview
Dự án này giải quyết bài toán dự đoán giá bất động sản (Regression) bằng cách kết hợp các đặc trưng kiến trúc truyền thống và **đặc trưng không gian (Geospatial Features)**. Mục tiêu là chứng minh việc áp dụng tọa độ địa lý (Latitude/Longitude) và công thức tính toán khoảng cách thực tế sẽ cải thiện độ chính xác của các mô hình Tree-based (XGBoost, LightGBM).

## 🎯 Research Questions (RQs)
1. **Khám phá Đặc trưng:** Đặc trưng không gian (khoảng cách đến trung tâm, tiện ích) ảnh hưởng như thế nào đến giá trị bất động sản so với các đặc trưng vật lý?
2. **So sánh Mô hình:** Các mô hình Tree-based (XGBoost, LightGBM) thể hiện hiệu năng ra sao khi xử lý dữ liệu không gian so với Linear/Ridge Regression?
3. **Đo lường Hiệu suất:** Việc tinh chỉnh siêu tham số (Hyperparameter tuning) giúp giảm sai số tuyệt đối trung bình (MAE) và gốc sai số toàn phương trung bình (RMSE) bao nhiêu phần trăm so với baseline?

## 🛠 Tech Stack
* **Database & Query:** SQLite / MySQL (Sử dụng Window Functions)
* **Data Processing:** `pandas`, `numpy`, công thức `Haversine` (Tính khoảng cách bề mặt cầu)
* **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`
* **Visualization:** `matplotlib`, `seaborn`, `folium`
* **Reporting:** LaTeX (Springer Template)

## 🚀 Pipeline & Execution
* **Tuần 1-2:** Ingestion dữ liệu, xử lý tọa độ, tính toán khoảng cách Haversine và phân tích dữ liệu bằng SQL.
* **Tuần 3-4:** Trực quan hóa dữ liệu không gian (Heatmap) và huấn luyện 5 Baseline Models.
* **Tuần 5-7:** Tối ưu hóa mô hình, xây dựng Dashboard dự đoán và soạn thảo báo cáo học thuật bằng LaTeX.

## 🧠 AI Reflection & Human Delta
Dự án này áp dụng phương pháp kiểm soát AI nghiêm ngặt thông qua quá trình ghi nhận **AI Audit Log**. Các dấu ấn can thiệp của con người (Human Delta) bao gồm:
* **Hiệu chỉnh thuật toán không gian:** Tự động bác bỏ đề xuất dùng khoảng cách Euclid 2D của AI để tính tọa độ, thay thế bằng công thức Haversine để tính chính xác khoảng cách trên mặt cầu Trái Đất.
* **Lựa chọn Metric chuyên sâu:** Nhận diện và sửa lỗi Hallucination khi AI đề xuất các metric của bài toán Phân loại (Accuracy, F1-Score) cho bài toán Hồi quy, chốt sử dụng MAE và RMSE.
* **Tối ưu SQL:** Chủ động tối ưu hóa các Subquery lồng nhau do AI tạo ra bằng cách sử dụng `Window Functions`, giúp giảm đáng kể độ phức tạp thuật toán.
