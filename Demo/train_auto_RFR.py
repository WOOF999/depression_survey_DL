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

autoencoder.save('autoencoder_model.h5')
# 그래프로 loss 확인'''
'''plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()'''


# Encoder 부분 추출
encoder = Model(autoencoder.input, autoencoder.layers[2].output)

# Encoder를 사용하여 데이터 인코딩
X_train_encoded = encoder.predict(X_train_scaled)
X_test_encoded = encoder.predict(X_test_scaled)

# Encoder를 사용하여 데이터 인코딩
#X_train_encoded = X_train_scaled
#X_test_encoded = X_test_scaled

# RandomForestRegressor 모델 정의

rf_model=RandomForestRegressor(n_estimators = 50, max_depth=6, min_samples_leaf=4, min_samples_split=8, n_jobs=-1, random_state=0)
'''from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators':[50,100,120,150,180],
    'max_depth' : [4,6,8], 
    'min_samples_leaf' : [4,6,8],
    'min_samples_split' : [6,8,10]
}

rf_clf = RandomForestRegressor(random_state=0, n_jobs=-1)
grid_cv = GridSearchCV(rf_clf , param_grid=params , cv=2, n_jobs=2, verbose=2 )
grid_cv.fit(X_train_encoded , y_train) # grid.cv.fit(train_x, train_y)

estimator =grid_cv.best_estimator_ 
pred = estimator.predict(X_test_encoded) # estimator.predict(test)

print('최적 하이퍼 파라미터:\n', grid_cv.best_params_)
print('최고 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))
mean_squared_error(y_test , pred)
'''

# RandomForestRegressor 모델 훈련
rf_model.fit(X_train_encoded, y_train)

# 예측
y_pred_rf = rf_model.predict(X_test_encoded)


# 평가
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"Random Forest MSE: {mse_rf}")

# 모델 저장
rf_filename = f'Saved_model/RFR/RFR_mse_{mse_rf:.4f}.joblib'
joblib.dump(rf_model, rf_filename)

# rf 모델 예측함수
def rf_predict(X):
    encoded_data = encoder.predict(scaler.transform(X))
    #encoded_data = scaler.transform(X)

    rf_prediction = rf_model.predict(encoded_data)
    return rf_prediction


# 새로운 데이터에 대한 예측
new_data = pd.DataFrame([[3,0,1,0,1,6,4,1,3,3,4,3,3,3,0,1,1,4,4,4,1,1,1,1,0,0,1,0]])
final_prediction = rf_predict(new_data)

#1

print(f'Predicted depression_total: {final_prediction[0]}')
