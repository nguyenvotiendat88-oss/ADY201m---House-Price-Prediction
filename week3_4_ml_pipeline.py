import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore') # Ẩn các cảnh báo phiên bản thư viện

def run_ml_pipeline(data_path):
    print("Đang đọc dữ liệu sạch...")
    df = pd.read_csv(data_path)
    
    # 1. Trực quan hóa dữ liệu (Tuần 3)
    print("Đang tạo biểu đồ phân bố giá nhà theo khoảng cách (EDA)...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Distance_to_Center_km', y='SalePrice', alpha=0.6, color='darkcyan')
    plt.title('House Prices vs. Distance to City Center')
    plt.xlabel('Distance to Center (km)')
    plt.ylabel('Sale Price (USD)')
    plt.grid(True)
    plt.savefig('spatial_price_distribution.png')
    print("Đã lưu biểu đồ: spatial_price_distribution.png\n")
    
    # 2. Lựa chọn Đặc trưng & Phân chia Dữ liệu (Tuần 4)
    features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'Distance_to_Center_km']
    X = df[features]
    y = df['SalePrice']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Khởi tạo 5 Mô hình Học máy
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
    }
    
    # 4. Huấn luyện và Đánh giá Metrics
    print("--- KẾT QUẢ HUẤN LUYỆN 5 MÔ HÌNH ---")
    print(f"{'Mô hình':<20} | {'MAE':<12} | {'RMSE':<12} | {'R2 Score':<8}")
    print("-" * 60)
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name:<20} | {mae:<12,.2f} | {rmse:<12,.2f} | {r2:.4f}")

if __name__ == "__main__":
    try:
        run_ml_pipeline('cleaned_house_prices.csv')
        print("\nHoàn tất Tuần 3 & 4. Sẵn sàng báo cáo số liệu!")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'cleaned_house_prices.csv'.")