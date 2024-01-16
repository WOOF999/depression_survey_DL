import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import joblib


# 데이터 로드
df = pd.read_csv('Data/survey_data_train_lv.csv')

# 데이터 전처리
X = df.drop('DE_LEVEL', axis=1)
X=df.drop('depression_total',axis=1)
y = df['DE_LEVEL']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Autoencoder 모델 정의
input_dim = X_train_scaled.shape[1]

autoencoder = Sequential([
    Dense(256, activation='relu', input_dim=input_dim),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(input_dim, activation='linear')
])

#autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.compile(optimizer=Adam(lr=0.0005), loss='mean_squared_error')

# Autoencoder 모델 훈련
history = autoencoder.fit(X_train_scaled, X_train_scaled, epochs=100, batch_size=32, shuffle=True, validation_data=(X_test_scaled, X_test_scaled))
x_pred = autoencoder.predict(X_test_scaled)
mse_auto= mean_squared_error(X_test_scaled, x_pred)

print(f"Autoencoder MSE: {mse_auto}")

# 그래프로 loss 확인'''
'''plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()'''

#autoencoder 모델 저장
autoencoder_filename = f'Saved_model/autoencoder/autoencoder_DL_mse_{mse_auto:.4f}.h5'
autoencoder.save(autoencoder_filename)