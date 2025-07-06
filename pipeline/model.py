import os
import logging
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import sys

# Set up logs folder and logging
logs_dir = 'logs'
os.makedirs(logs_dir, exist_ok=True)
log_file_path = os.path.join(logs_dir, 'training.log')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)

class PMModelTrainer:
    def __init__(self, data_path: str, model_dir: str):
        self.data_path = data_path
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def load_data(self):
        df = pd.read_csv(self.data_path)
        if 'datetime' in df.columns:
            df = df.drop(columns=['datetime'])
        df = df.dropna()
        self.X = df.drop(columns=['PM2.5', 'PM10'], errors='ignore')
        self.y_pm25 = df['PM2.5']

    def plot_predicted_vs_actual(self, model, X_test, y_test, target):
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue', edgecolors='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{target} Prediction\nRMSE: {rmse:.2f}')
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(self.model_dir, f'{target}_scatter.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f'{target} scatter plot saved to {plot_path}')

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y_pm25, test_size=0.2, random_state=42)

        param_grid = {
                'n_estimators': [100, 300, 500],             # More trees generally improve stability
                'max_depth': [None, 10, 20, 30],             # None = expand fully, try limiting depth
                'min_samples_split': [2, 5, 10],             # Higher = more regularization
                'min_samples_leaf': [1, 2, 4],               # Higher = smoother, prevents overfitting
                'max_features': ['sqrt', 'log2', None]       # None = all features, 'sqrt' is default
            }


        rf = RandomForestRegressor(random_state=42, n_jobs=-1)

        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,
            n_jobs=-1,
            verbose=2,
            scoring='r2'
        )

        logging.info("Starting GridSearchCV for PM2.5...")
        grid_search.fit(X_train, y_train)

        # Save grid search results
        results_df = pd.DataFrame(grid_search.cv_results_)
        csv_path = os.path.join(self.model_dir, 'PM2.5_grid_search_results.csv')
        results_df.to_csv(csv_path, index=False)
        logging.info(f"Saved GridSearchCV results to {csv_path}")

        best_model = grid_search.best_estimator_
        logging.info(f"Best parameters for PM2.5: {grid_search.best_params_}")

        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logging.info(f'PM2.5 model MSE: {mse:.4f}')
        logging.info(f'PM2.5 model RMSE: {rmse:.4f}')
        logging.info(f'PM2.5 model MAE: {mae:.4f}')
        logging.info(f'PM2.5 model R²: {r2:.4f}')

        joblib.dump(best_model, os.path.join(self.model_dir, 'rf_model_pm2.5.pkl'))

        self.plot_predicted_vs_actual(best_model, X_test, y_test, 'PM2.5')
        self.plot_residuals(y_test, y_pred, 'PM2.5')

    def plot_residuals(self, y_true, y_pred, target):
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 5))
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f'Residuals for {target}')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, f'residuals_{target.lower()}.png'))
        plt.close()

    def execute(self):
        self.load_data()
        self.train_model()

if __name__ == '__main__':
    trainer = PMModelTrainer(
        data_path='data/model_input_dataset.csv',
        model_dir='models/random_forest_9'
    )
    trainer.execute()
