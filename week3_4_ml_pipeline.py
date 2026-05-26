import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("Bắt đầu load data ")
# Đọc file csv sạch đợt trước tạo ra
df = pd.read_csv('cleaned_house_prices.csv')

# Khai báo cột X với Y
X = df[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'Distance_to_Center_km']]
Y = df['SalePrice']

# Chia 80/20. random_state = 42 để không bị đổi kết quả
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# scale data không là AI nó sẽ lỗi vì diện tích to quá
cai_can = StandardScaler()
x_train_scale = cai_can.fit_transform(x_train)
x_test_scale = cai_can.transform(x_test) # test thì transform thôi

print("\nĐang train 5 mô hình ")

# ----------------------------------------------------
# 1. Hồi quy tuyến tính (Linear Regression)
print("--- 1. Linear Regression ---")
model_1 = LinearRegression()
model_1.fit(x_train_scale, y_train)
kq_1 = model_1.predict(x_test_scale)
print("Sai số MAE: ", mean_absolute_error(y_test, kq_1))
print("Điểm R2: ", r2_score(y_test, kq_1))

# ----------------------------------------------------
# 2. Cây quyết định (Decision Tree)
print("--- 2. Decision Tree ---")
model_2 = DecisionTreeRegressor(random_state=42)
model_2.fit(x_train_scale, y_train)
kq_2 = model_2.predict(x_test_scale)
print("Sai số MAE: ", mean_absolute_error(y_test, kq_2))
print("Điểm R2: ", r2_score(y_test, kq_2))

# ----------------------------------------------------
# 3. Rừng ngẫu nhiên (Random Forest) 
print("--- 3. Random Forest ---")
model_3 = RandomForestRegressor(n_estimators=100, random_state=42)
model_3.fit(x_train_scale, y_train)
kq_3 = model_3.predict(x_test_scale)
print("Sai số MAE: ", mean_absolute_error(y_test, kq_3))
print("Điểm R2: ", r2_score(y_test, kq_3))

# ----------------------------------------------------
# 4. Gradient Boosting
print("--- 4. Gradient Boosting ---")
model_4 = GradientBoostingRegressor(n_estimators=100, random_state=42)
model_4.fit(x_train_scale, y_train)
kq_4 = model_4.predict(x_test_scale)
print("Sai số MAE: ", mean_absolute_error(y_test, kq_4))
print("Điểm R2: ", r2_score(y_test, kq_4))

# ----------------------------------------------------
# 5. XGBoost (Thằng này xịn nhất để làm web nè)
print("--- 5. XGBoost ---")
model_5 = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model_5.fit(x_train_scale, y_train)
kq_5 = model_5.predict(x_test_scale)
print("Sai số MAE: ", mean_absolute_error(y_test, kq_5))
print("Điểm R2: ", r2_score(y_test, kq_5))

# ----------------------------------------------------
print("\nTrain xong rồi, giờ lưu model 5 (XGBoost) với cái cân ra file pkl...")
joblib.dump(model_5, 'best_xgboost_model.pkl')
joblib.dump(cai_can, 'scaler.pkl')

print("Oke lưu xong! Test thử giao diện web được rồi nha!")