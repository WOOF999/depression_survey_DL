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
from keras.models import load_model
import numpy as np

# 데이터 로드
df = pd.read_csv('Data/survey_data_train.csv')

# 데이터 전처리
X = df.drop('depression_total', axis=1)
y = df['depression_total']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# pretrained autoencoder load
autoencoder = load_model('Saved_model/autoencoder/autoencoder_mse_0.1185.h5')

# Encoder 부분 추출
encoder = Model(autoencoder.input, autoencoder.layers[2].output)

# Encoder를 사용하여 데이터 인코딩
X_train_encoded = encoder.predict(X_train_scaled)
X_test_encoded = encoder.predict(X_test_scaled)


# RandomForestRegressor 모델 정의
rf_model=RandomForestRegressor(n_estimators = 50, max_depth=6, min_samples_leaf=4, min_samples_split=8, n_jobs=-1, random_state=0)


#최적 parameter 찾기
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

# 정수로 변환
y_pred_rf_int = np.round(y_pred_rf).astype(int)
# 또는ed_rf)
# y_pred_rf_int = y_pred_rf.astype(int)

# 평가
mse_rf = mean_squared_error(y_test, y_pred_rf_int)
print(f"Random Forest MSE: {mse_rf}")

# 모델 저장
rf_filename = f'Saved_model/RFR/RFR_mse_{mse_rf:.4f}.joblib'
joblib.dump(rf_model, rf_filename)
