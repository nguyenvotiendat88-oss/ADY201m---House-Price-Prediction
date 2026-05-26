import sqlite3
import pandas as pd
import numpy as np
import math

# Hàm tính khoảng cách lấy trên mạng (Công thức Haversine)
def tinh_khoang_cach(lat1, lon1, lat2, lon2):
    # Đổi sang radian hết để dùng hàm sin cos
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    return c * 6371 # 6371 là bán kính trái đất (km)

# Tọa độ trung tâm thành phố Ames
tam_lat = 42.0308
tam_lon = -93.6319

# Tọa độ các khu vực (Tra cứu thủ công trên Google Maps)
toado_khu = {
    'CollgCr': (42.0187, -93.6855), 'Veenker': (42.0401, -93.6570),
    'Crawfor': (42.0254, -93.6423), 'NoRidge': (42.0503, -93.6560),
    'Mitchel': (41.9901, -93.6009), 'Somerst': (42.0521, -93.6434),
    'NWAmes': (42.0458, -93.6305), 'OldTown': (42.0288, -93.6155),
    'BrkSide': (42.0324, -93.6260), 'Sawyer': (42.0336, -93.6821),
    'NridgHt': (42.0603, -93.6550), 'NAmes': (42.0429, -93.6135),
    'SawyerW': (42.0355, -93.6850), 'IDOTRR': (42.0220, -93.6235),
    'MeadowV': (41.9918, -93.6025), 'Edwards': (42.0228, -93.6630),
    'Timber': (41.9986, -93.6525), 'Gilbert': (42.1080, -93.6496),
    'StoneBr': (42.0602, -93.6355), 'ClearCr': (42.0254, -93.6750),
    'NPkVill': (42.0502, -93.6250), 'Blmngtn': (42.0628, -93.6390),
    'BrDale': (42.0525, -93.6288), 'SWISU': (42.0225, -93.6450),
    'Blueste': (42.0100, -93.6450)
}

# Hàm phụ để apply vào cột của dataframe
def tinh_km(ten_khu):
    if ten_khu in toado_khu:
        lat, lon = toado_khu[ten_khu]
        return tinh_khoang_cach(lat, lon, tam_lat, tam_lon)
    else:
        # Nếu có khu nào bị rớt không tìm thấy thì cho đại 5km để code không bị lỗi
        return 5.0 

def chay_data():
    print("Đang đọc data từ file train.csv...")
    df = pd.read_csv('train.csv')

    # Dùng SQL nên tạo một database ảo trên RAM
    conn = sqlite3.connect(':memory:')
    df.to_sql('nha_dat', conn, index=False)

    # Viết câu query lọc rác và lấy mấy cột cần xài
    chuoi_sql = """
    SELECT OverallQual, GrLivArea, GarageCars, TotalBsmtSF, Neighborhood, SalePrice
    FROM nha_dat
    WHERE OverallQual IS NOT NULL 
      AND SalePrice IS NOT NULL
      AND GrLivArea > 0
    """
    
    # Kéo data từ SQL ra lại Pandas
    df_moi = pd.read_sql_query(chuoi_sql, conn)
    conn.close()

    print("Đang tính khoảng cách từ nhà đến trung tâm...")
    # Dùng hàm apply để tính km cho từng dòng
    df_moi['Distance_to_Center_km'] = df_moi['Neighborhood'].apply(tinh_km)

    # Lọc lại đúng 6 cột đem đi train, quét dropna lần cuối cho chắc ăn
    cot_can_lay = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'Distance_to_Center_km', 'SalePrice']
    df_clean = df_moi[cot_can_lay].dropna()
    
    # Lưu ra file csv sạch
    df_clean.to_csv('cleaned_house_prices.csv', index=False)
    print(f" Done! Đã lưu {len(df_clean)} dòng vào file cleaned_house_prices.csv")

if __name__ == "__main__":
    chay_data()