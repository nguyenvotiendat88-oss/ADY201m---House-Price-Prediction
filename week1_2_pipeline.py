import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

# 1. Hàm tính khoảng cách bề mặt Trái Đất (Haversine)
def haversine_distance(lat1, lon1, lat2, lon2):
    """Tính khoảng cách (km) giữa 2 điểm địa lý."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Bán kính Trái Đất (km)
    return c * r

# 2. Xử lý dữ liệu
def process_data(file_path):
    df = pd.read_csv(file_path)
    
    # Điền giá trị thiếu (Missing Values) cơ bản
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # Tọa độ giả định cho các khu vực (Neighborhood) ở Ames, Iowa
    coords = {
        'CollgCr': (42.0313, -93.6526), 'Veenker': (42.0401, -93.6515), 
        'Crawfor': (42.0254, -93.6429), 'NoRidge': (42.0503, -93.6560),
        'Mitchel': (41.9901, -93.6009), 'Somerst': (42.0521, -93.6434)
    }
    
    # Gán tọa độ (Nếu không có thì lấy mặc định trung tâm Ames)
    df['Latitude'] = df['Neighborhood'].map(lambda x: coords.get(x, (42.0345, -93.6204))[0])
    df['Longitude'] = df['Neighborhood'].map(lambda x: coords.get(x, (42.0345, -93.6204))[1])
    
    # Tọa độ trung tâm thành phố (City Center)
    center_lat, center_lon = 42.0345, -93.6204
    
    # Áp dụng hàm Haversine tạo Đặc trưng Không gian
    df['Distance_to_Center_km'] = df.apply(
        lambda row: haversine_distance(row['Latitude'], row['Longitude'], center_lat, center_lon), axis=1
    )
    
    return df

if __name__ == "__main__":
    print("Đang xử lý dữ liệu...")
    try:
        df_clean = process_data('train.csv')
        df_clean.to_csv('cleaned_house_prices.csv', index=False)
        print(f"Hoàn tất Tuần 1 & 2! Kích thước dữ liệu sạch: {df_clean.shape}")
        print("Đã lưu file: cleaned_house_prices.csv")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file 'train.csv'.")