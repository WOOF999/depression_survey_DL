import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

# 데이터 불러오기
data = pd.read_csv('survey_data_train.csv')

# X와 y로 데이터 분리
X = data.drop('depression_total', axis=1)
y = data['depression_total']

# 표준화 (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 학습 데이터와 테스트 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# MLP 모델 정의
mlp = MLPRegressor(hidden_layer_sizes=(150, 100, 50),
                   max_iter=1000, activation='relu',
                   solver='adam', random_state=42)

# Lists to store training and validation losses
train_losses = []
val_losses = []

# Number of iterations
n_iter = 1000

for i in range(n_iter):
    # Partial fit for each iteration
    mlp.partial_fit(X_train, y_train)

    # Calculate training loss
    train_loss = mlp.loss_
    train_losses.append(train_loss)

    # Calculate validation loss
    y_val_pred = mlp.predict(X_test)
    val_loss = mean_squared_error(y_test, y_val_pred)
    val_losses.append(val_loss)

# Plotting the loss curves
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title("Training and Validation Loss Curves", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 테스트 데이터에 대한 예측
y_pred = mlp.predict(X_test)

# 평가 지표 출력 (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
