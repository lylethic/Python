import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Bước 1: Đọc dữ liệu từ File Excel và làm sạch dữ liệu
url=('yahoo_data[1].xlsx')

# dropna() => Chọn các cột cần thiết và loại bỏ giá trị thiếu (xóa các cột NaN)
data = pd.read_excel(url)[['Open', 'High', 'Low', 'Close*', 'Volume']].dropna()

# Hiển thị các thông tin ban đầu của dữ liệu
# Kiểm tra thông tin ban đầu và cấu trúc dữ liệu (số lượng cột, kiểu dữ liệu,...).
print(data.head())
print(data.info())

# Bước 2: Tiền xử lý dữ liệu
# Giả sử rằng cột 'Close*' là biến mục tiêu và các cột còn lại là biến độc lập
# Tách đặc trưng biến độc lập và phụ thuộc
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close*']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Chuẩn hóa dữ liệu
# tools giúp cho các đặc trưng ( X_train ) có trung bình bằng 0 và lệch chuẩn = 1,
# Sẽ chuyển thành số dạng ( -1 < 0 < 1 )
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Bước 3: Xây dựng mô hình hồi quy tuyến tính với Gradient Descent
# Trong đó:
#   a (alpha): Hay còn gọi là tốc độ học (learning rate), 
#               là hệ số điều chỉnh mức độ thay đổi của các trọng số sau mỗi lần cập nhật.
#   iterations: Số lần lặp của thuật toán Gradient Descent.
#   cost_history: Lưu trữ lịch sử của hàm chi phí trong quá trình huấn luyện để theo dõi mức độ hội tụ.
class LinearRegressionGD:
    def __init__(self, a = 0.01, iterations = 1000):
        self.a = a
        self.iterations = iterations
        self.cost_history = []

    # Hàm dùng huấn luyện mô hình
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) # trọng số (weights) tương ứng với từng đặc trưng
        self.bias = 0 # bias dịch chuyển đường hồi quy hay hằng số chặn

        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.a * dw
            self.bias -= self.a * db

            # Lưu lịch sử hàm chi phí
            cost = np.mean((y_pred - y) ** 2) / 2
            self.cost_history.append(cost)

# dự đoán giá trị đầu ra cho các dữ liệu mới dựa trên trọng số và độ dời (bias) đã học.
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Bước 4: Khởi tạo và huấn luyện mô hình
model = LinearRegressionGD(a = 0.01, iterations = 1000)
model.fit(X_train, y_train)

# Dự đoán và đánh giá
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Tính toán MSE và R2 => result = số thập phân
# MSE: Trung bình bình phương sai số giữa giá trị thực và giá trị dự đoán.
print(f'Training MSE: {mean_squared_error(y_train, y_train_pred):.4f}')
print(f'Test MSE: {mean_squared_error(y_test, y_test_pred):.4f}')
# Đo lường mức độ giải thích được của mô hình, 
# chỉ ra bao nhiêu phần trăm biến thiên của dữ liệu có thể được giải thích bởi mô hình.
print(f'Training R2: {r2_score(y_train, y_train_pred):.4f}')
print(f'Test R2: {r2_score(y_test, y_test_pred):.4f}')

# Bước 5: Trực quan hóa kết quả
plt.figure(figsize = (10, 6))
plt.scatter(y_test, y_test_pred, color = 'blue', alpha = 0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2)
plt.title('Giá Thực tế & Giá Dự đoán')
plt.xlabel('Giá thực tế')
plt.ylabel('Giá dự đoán')
plt.show()

# Bước 6: Trực quan hóa hàm chi phí
plt.figure(figsize=(10, 5))
plt.plot(model.cost_history, color='orange')
plt.title('Quá trình hội tụ của Gradient Descent')
plt.xlabel('Số lần lặp')
plt.ylabel('Hàm chi phí (Cost)')
plt.grid(True)
plt.show()