import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Obyekt yaratish
class AnomalyDetector:
    def __init__(self, data):
        self.data = data

    # O'zgaruvchilarni o'rganish
    def feature_engineering(self):
        # O'zgaruvchilarni o'rganish
        return self.data

    # O'zgaruvchilarni mustahkamlash
    def feature_scaling(self):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data)
        return scaled_data

    # O'zgaruvchilar uchun modelni tayyorlash
    def model_tayyorlash(self, scaled_data):
        X = scaled_data[:, :-1]
        y = scaled_data[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = IsolationForest(n_estimators=100, contamination=0.1)
        model.fit(X_train)
        return model, X_test, y_test

    # Modelni qo'llash
    def model_qo'llash(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    # O'zgaruvchilar uchun modelni qo'llash
    def model_qo'llash_uzlar(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

# Test ma'lumotlari
data = pd.DataFrame({
    'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'B': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'C': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
})

# Anomaly Detector yaratish
detector = AnomalyDetector(data)

# O'zgaruvchilar uchun o'rganish
scaled_data = detector.feature_scaling()

# Modelni tayyorlash
model, X_test, y_test = detector.model_tayyorlash(scaled_data)

# Modelni qo'llash
accuracy = detector.model_qo'llash(model, X_test, y_test)
print("Modelning to'g'ri ishlashi:", accuracy)
