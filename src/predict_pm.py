import joblib
import pandas as pd

class PMModelPredictor:
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
        pm25_pred = self.model_pm25.predict(features)[0]
        pm10_pred = self.model_pm10.predict(features)[0]
        return round(pm25_pred, 2), round(pm10_pred, 2)

if __name__ == '__main__':
    predictor = PMModelPredictor(
        model_pm25_path='models/random_forest/rf_model_pm2.5.pkl',
        model_pm10_path='models/random_forest/rf_model_pm10.pkl'
    )

    aod = None
    HFLUX = 154.0964203
    PBLH = 971.8314819
    QLML = 0.003878968
    SPEED = 1.334718823
    TLML = 1.334718823

    pm25, pm10 = predictor.predict(aod, HFLUX, PBLH, QLML, SPEED, TLML)
    print(f"Predicted PM2.5: {pm25}")
    print(f"Predicted PM10: {pm10}")
