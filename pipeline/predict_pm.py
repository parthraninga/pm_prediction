import joblib
import pandas as pd

class PM10ModelPredictor:
    def __init__(self, model_pm25_path, model_pm10_path):
        self.model_pm25 = joblib.load(model_pm25_path)
        self.model_pm10 = joblib.load(model_pm10_path)

    def predict(self, aod, HFLUX, PBLH, QLML, SPEED, TLML):
        features = pd.DataFrame([{
            'aod': aod,
            'HFLUX': HFLUX,
            'PBLH': PBLH,
            'QLML': QLML,
            'SPEED': SPEED,
            'TLML': TLML
        }])

        # Predict PM2.5
        pm25_pred = self.model_pm25.predict(features)[0]
        features.insert(loc=features.columns.get_loc('aod') + 1, column='PM2.5', value=pm25_pred)

        # Predict PM10 using PM2.5 + other features
        pm10_pred = self.model_pm10.predict(features)[0]

        return round(pm25_pred, 2), round(pm10_pred, 2)

if __name__ == '__main__':
    predictor = PM10ModelPredictor(
        model_pm25_path='models/random_forest_9/rf_model_pm2.5.pkl',
        model_pm10_path='models/pm10_from_pm25_xgb/xgb_model_pm10.pkl'
    )

    aod = 0.094660684
    HFLUX = 154.0964203
    PBLH = 971.8314819
    QLML = 0.003878968
    SPEED = 1.334718823
    TLML = 294.7254333

    pm25, pm10 = predictor.predict(aod, HFLUX, PBLH, QLML, SPEED, TLML)
    print(f"Predicted PM2.5: {pm25}")
    print(f"Predicted PM10: {pm10}")
