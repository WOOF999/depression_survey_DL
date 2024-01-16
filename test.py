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

# 데이터 표준화
scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)


# model 불러옴
autoencoder=load_model('Saved_model/autoencoder/autoencoder_vif_mse_0.0889.h5')
rf_model=joblib.load("Saved_model/RFC/RFC_accuracy_0.4615.joblib")

encoder = Model(autoencoder.input, autoencoder.layers[2].output)

# rf 모델 예측함수
def rf_predict(X):
    encoded_data = encoder.predict(scaler.fit_transform(X))
    rf_prediction = rf_model.predict(encoded_data)
    return rf_prediction


# 새로운 데이터에 대한 예측 5
new_data = pd.DataFrame([[2,5,2,4,5,4,1,0,7,4,1,3,2,0,0,0,0,0,5,20]])
final_prediction = rf_predict(new_data)


print(f'Predicted depression_total: {final_prediction[0]}')
