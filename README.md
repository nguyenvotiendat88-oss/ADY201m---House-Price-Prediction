🏡 House Price Prediction with Geospatial Features
Course: ADY201m - AI & Data Science with Python & SQL
University: FPT University
Dataset: Ames Housing (Kaggle) supplemented with Geospatial Data

📌 Project Overview
Dự án xây dựng một hệ thống Data Pipeline khép kín để dự đoán giá bất động sản. Khác biệt cốt lõi của dự án là việc loại bỏ cách phân tích chỉ dựa trên đặc trưng vật lý truyền thống, thay vào đó tích hợp Đặc trưng không gian (Geospatial Features). Hệ thống bao gồm quy trình truy xuất bằng SQLite in-memory, huấn luyện qua 5 mô hình học máy (XGBoost là lõi) và triển khai thực tế bằng Web Dashboard Streamlit.

🎯 Research Questions (RQs)
Khám phá Đặc trưng: Khoảng cách địa lý tác động thế nào đến giá nhà so với các thuộc tính vật lý (diện tích, chất lượng)?

Đánh giá Mô hình: Trong số 5 thuật toán học máy (Linear, Decision Tree, Random Forest, Gradient Boosting, XGBoost), mô hình nào cân bằng tốt nhất giữa độ chính xác và khả năng triển khai thực tế?

Phân tích Thiên kiến (Data Bias): Tư duy con người (Human Delta) có thể giúp phát hiện những nghịch lý về quy hoạch văn hóa mà cỗ máy AI không thể tự hiểu như thế nào?

🛠 Tech Stack
Database: SQLite3 (In-memory Database)

Data Processing: pandas, numpy, thuật toán Haversine

Machine Learning: scikit-learn, xgboost, joblib

Web Deployment: streamlit

Reporting: LaTeX (PDF Export)

🚀 Cấu trúc Hệ thống (Pipeline)
Dự án được chia module rành mạch, đảm bảo tính cô lập và dễ bảo trì:

1. Xử lý Dữ liệu (week1_2_pipeline.py): Khởi tạo cơ sở dữ liệu SQLite ảo, làm sạch dữ liệu bằng SQL, tính toán khoảng cách bề mặt cầu bằng công thức Haversine, xuất ra file cleaned_house_prices.csv.

2. Huấn luyện AI (week3_4_ml_pipeline.py): Chuẩn hóa dữ liệu (StandardScaler). Huấn luyện và đánh giá 5 mô hình. Khóa băng (freeze) mô hình XGBoost tốt nhất (R² = 88%) ra các file nhị phân best_xgboost_model.pkl và scaler.pkl.

3. Triển khai Web (week5_dashboard.py): Xây dựng giao diện tương tác, load mô hình từ file PKL, tự động quy đổi giá trị dự đoán từ USD sang VNĐ.

🧠 AI Reflection & Human Delta
Dự án được kiểm soát nghiêm ngặt bằng logic của kỹ sư thay vì phụ thuộc 100% vào AI. Các dấu ấn can thiệp của con người (Human Delta) bao gồm:

Xử lý Thiên kiến Dữ liệu (Data Bias): AI tìm ra quy luật "Nhà càng xa trung tâm, giá càng cao". Máy móc chỉ nhìn vào số liệu, nhưng tư duy con người (Human Delta) đã giải thích được nghịch lý này: Đây là văn hóa quy hoạch vùng ngoại ô (Suburban) của Mỹ (giới tinh hoa thích ở xa lõi đô thị), hoàn toàn trái ngược với thị trường bất động sản lõi trung tâm tại Việt Nam.

Hiệu chỉnh thuật toán không gian: Bác bỏ việc sử dụng khoảng cách Euclid 2D cơ bản, chủ động áp dụng công thức lượng giác Haversine để tính chính xác khoảng cách cung tròn trên mặt cầu Trái Đất nhằm giảm thiểu tối đa sai số địa lý.

Đánh đổi trong Model Selection: Mặc dù Gradient Boosting có điểm R² cao hơn một chút (89.7%), kỹ sư chủ động chọn XGBoost (88.0%) làm mô hình Production vì kiến trúc của XGBoost xử lý ngoại lệ mạnh mẽ hơn, tốc độ suy luận (inference speed) nhanh hơn, phù hợp để đưa lên Web Dashboard.
