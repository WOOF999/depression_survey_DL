import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

# 데이터 로드
df = pd.read_csv('survey_data_train.csv')

# 데이터 전처리
X = df.drop('depression_total', axis=1)
y = df['depression_total']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Autoencoder 모델 정의
input_dim = X_train_scaled.shape[1]

autoencoder = Sequential([
    Dense(64, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(input_dim, activation='linear', kernel_regularizer=l2(0.01))
])

autoencoder.compile(optimizer=Adam(lr=0.0005), loss='mean_squared_error')

# Autoencoder 모델 훈련
history = autoencoder.fit(X_train_scaled, X_train_scaled, epochs=500, batch_size=32, shuffle=True, validation_data=(X_test_scaled, X_test_scaled))
autoencoder.save('autoencoder_model.h5')

# 그래프로 loss 확인
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Encoder 부분 추출
encoder = Model(autoencoder.input, autoencoder.layers[2].output)

# Encoder를 사용하여 데이터 인코딩
X_train_encoded = encoder.predict(X_train_scaled)
X_test_encoded = encoder.predict(X_test_scaled)

# Semi-supervised fine-tuning을 위한 모델 정의
model = Sequential()
model.add(encoder)  # Autoencoder에서 얻은 Encoder를 첫 번째 레이어로 추가
model.add(Dropout(0.5))  # Add dropout layer
model.add(Dense(1, activation='linear', kernel_regularizer=l2(0.01)))  # Regression을 위한 출력 레이어 with L2 regularization

# Fine-tuning 모델 컴파일
model.compile(optimizer=Adam(lr=0.0008), loss='mean_squared_error')

# Semi-supervised fine-tuning 모델 학습
history_fine_tuning = model.fit(X_train_scaled, y_train, epochs=500, batch_size=32, shuffle=True, validation_data=(X_test_scaled, y_test))

# 그래프로 fine-tuning loss 확인
plt.plot(history_fine_tuning.history['loss'], label='Train Loss (Fine-tuning)')
plt.plot(history_fine_tuning.history['val_loss'], label='Validation Loss (Fine-tuning)')
plt.title('Fine-tuning Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 모델 평가
y_pred_fine_tuned = model.predict(X_test_scaled)
mse_fine_tuned = mean_squared_error(y_test, y_pred_fine_tuned)
print(f"Fine-tuned Model MSE: {mse_fine_tuned}")

# 새로운 데이터에 대한 예측 17
new_data = pd.DataFrame([[4,0,1,0,1,7,3,0,1,5,4,2,5,4,0,0,7,4,1,3,0,0,2,1,0,0,1,0]])
final_prediction = model.predict(scaler.transform(new_data))
print(f'Predicted depression_total after fine-tuning: {final_prediction[0]}')