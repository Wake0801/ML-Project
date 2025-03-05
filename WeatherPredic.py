import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# Đọc dữ liệu
df = pd.read_csv("E:\Code\Ai\dataset\weather_prediction_dataset.csv")

# Chuyển đổi DATE thành datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Tạo cột nhiệt độ tối đa ngày mai
df['TOURS_temp_max_next_day'] = df['TOURS_temp_max'].shift(-1)

# Loại bỏ hàng cuối cùng vì không có giá trị ngày mai
df = df.iloc[:-1]

# Chọn các đặc trưng liên quan
drop_columns = ['DATE']  # Không cần DATE vì đã có thông tin thời gian
features = [col for col in df.columns if col not in drop_columns + ['TOURS_temp_max_next_day']]

df = df.dropna()  # Loại bỏ hàng có giá trị thiếu

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(df[features], df['TOURS_temp_max_next_day'], test_size=0.2, random_state=42)

# KNN Model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
mae_knn = mean_absolute_error(y_test, y_pred_knn)
print(f"KNN MAE: {mae_knn:.2f}")

# Điểm mạnh và điểm yếu của KNN
print("KNN Strengths: Dễ hiểu, hoạt động tốt với dữ liệu nhỏ, không cần giả định về phân phối dữ liệu.")
print("KNN Weaknesses: Chậm với dữ liệu lớn, dễ bị ảnh hưởng bởi nhiễu, cần chọn K phù hợp.")

# Moving Average Model
window_size = 7
df['Moving_Avg'] = df['TOURS_temp_max'].rolling(window=window_size).mean()

y_pred_ma = df['Moving_Avg'].iloc[-len(y_test):].values
y_pred_ma = np.nan_to_num(y_pred_ma, nan=np.mean(y_train))  # Thay NaN bằng trung bình
test_indices = ~np.isnan(y_test)
mae_ma = mean_absolute_error(y_test[test_indices], y_pred_ma[test_indices])
print(f"Moving Average MAE: {mae_ma:.2f}")

# Điểm mạnh và điểm yếu của Moving Average
print("Moving Average Strengths: Dễ triển khai, hiệu quả với dữ liệu có xu hướng ổn định.")
print("Moving Average Weaknesses: Không phản ứng nhanh với thay đổi đột ngột, không hoạt động tốt với dữ liệu có biến động cao.")

# Xuất tập dữ liệu đã tiền xử lý
processed_file = "E:\Code\Ai\dataset\weather_prediction_dataset.csv"
df.to_csv(processed_file, index=False)

print(f"Dữ liệu đã được tiền xử lý và lưu tại: {processed_file}")
