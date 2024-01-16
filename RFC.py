import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # 수정
from sklearn.metrics import accuracy_score  # 수정
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import joblib
from keras.models import load_model

# 데이터 로드
df = pd.read_csv('Data/train_vif_binary.csv')

# 데이터 전처리
X = df.drop(columns=['DE_LEVEL','Depression_Level_int','depression_total'])
y = df['DE_LEVEL']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# pretrained autoencoder load
autoencoder = load_model('Saved_model/autoencoder/AE_vif_binary_mse_0.0748.h5')

# Encoder 부분 추출
encoder = Model(autoencoder.input, autoencoder.layers[2].output)

# Encoder를 사용하여 데이터 인코딩
X_train_encoded = encoder.predict(X_train_scaled)
X_test_encoded = encoder.predict(X_test_scaled)

from sklearn.model_selection import GridSearchCV

params = { 'n_estimators' : [50,100,150],
           'max_depth' : [2,3,4,5],
           'min_samples_leaf' : [3,4,5,6],
           'min_samples_split' : [1,2,3,4,5]
            }

# RandomForestClassifier 객체 생성 후 GridSearchCV 수행
rf_clf = RandomForestClassifier(random_state = 0, n_jobs = -1)
grid_cv = GridSearchCV(rf_clf, param_grid = params, cv = 3, n_jobs = -1)
grid_cv.fit(X_train_encoded, y_train)

print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))


# RandomForestClassifier 모델 정의
rf_model = RandomForestClassifier(n_estimators=50, max_depth=3, min_samples_leaf=4, min_samples_split=2, n_jobs=-1, random_state=0)  # 수정
'''rf_model = RandomForestClassifier(n_estimators = 100, 
                                max_depth = 12,
                                min_samples_leaf = 8,
                                min_samples_split = 8,
                                random_state = 0,
                                n_jobs = -1)'''
rf_model.fit(X_train, y_train)
pred = rf_model.predict(X_test)
print('예측 정확도: {:.4f}'.format(accuracy_score(y_test,pred)))


# RandomForestClassifier 모델 훈련
rf_model.fit(X_train_encoded, y_train)

# 예측
y_pred_rf = rf_model.predict(X_test_encoded)

# 평가
accuracy_rf = accuracy_score(y_test, y_pred_rf)  # 수정
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

# 모델 저장
rf_filename = f'Saved_model/RFC/RFC_binary_ACC_{accuracy_rf:.4f}.joblib'  # 수정
joblib.dump(rf_model, rf_filename)
