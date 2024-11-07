import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Bước 1: Tải và làm sạch dữ liệu
url = './yahoo_data[1].xlsx'  
data = pd.read_excel(url)

# In các cột thông tin ban đầu của dữ liệu
print(data.head())
print(data.info())

# Bước 2: Tiền xử lý dữ liệu
# Giả sử rằng cột 'Close' là biến mục tiêu và các cột còn lại là biến độc lập
# Chọn các cột cần thiết và loại bỏ giá trị thiếu (xóa các cột NaN)
data = data[['Open', 'High', 'Low', 'Close*', 'Volume']].dropna()  

# Tách dữ liệu thành biến độc lập và phụ thuộc
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close*']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
# tools giúp cho các đặc trưng ( X_train ) có trung bình bằng 0 và lệch chuẩn = 1,
#  nôm na là sẽ chuyển thành số dạng ( -1 < 0 < 1 )
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)

# Bước 3: Xây dựng mô hình hồi quy tuyến tính với lưu trữ lịch sử hàm chi phí
class LinearRegressionGD:
    def __init__(self, learning_rate = 0.01, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []  # Thêm mảng để lưu trữ lịch sử của hàm chi phí

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent 
        for i in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Cập nhật trọng số và độ lệch
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Tính toán hàm chi phí (MSE) và lưu trữ
            cost = (1 / (2 * n_samples)) * np.sum((y_predicted - y) ** 2)
            self.cost_history.append(cost)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# Khởi tạo mô hình và huấn luyện
model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

# Bước 4: Dự đoán và đánh giá mô hình
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Tính toán MSE và R2 score
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# In kết quả
print(f'Training MSE: {mse_train}')
print(f'Test MSE: {mse_test}')
print(f'Training R2: {r2_train}')
print(f'Test R2: {r2_test}')

# Bước 5: Trực quan hóa kết quả dự đoán
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_test_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title('Giá thực tế vs Dự đoán')
plt.xlabel('Giá thực tế')
plt.ylabel('Giá dự đoán')
plt.show()

# Bước 6: Trực quan hóa lịch sử hàm chi phí
plt.figure(figsize=(10,6))
plt.plot(range(len(model.cost_history)), model.cost_history, color='blue')
plt.title('Quá trình hội tụ của Gradient Descent')
plt.xlabel('Số lần lặp (Iterations)')
plt.ylabel('Hàm chi phí (Cost function)')
plt.grid(True)
plt.show()
