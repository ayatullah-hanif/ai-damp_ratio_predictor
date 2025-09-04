🧱 Concrete Damping Ratio Prediction

This project predicts the damping ratio of concrete based on mix design parameters such as cement, sand, aggregate, rubber content, and water-to-cement ratio.
It uses a machine learning model (Random Forest Regressor) trained on experimental dataset to provide fast and accurate predictions.

🚀 Features

Predict damping ratio using six input parameters:

Cement content (kg/m³)

Fine aggregate (sand) content (kg/m³)

Coarse aggregate content (kg/m³)

Rubber content (kg/m³)

Water-to-cement ratio

Water (%)

Interactive Streamlit web app for user-friendly predictions.

Displays model performance metrics (RMSE, MAE, R², MAPE).

Visualizations for better model interpretation (Predicted vs Actual, Residuals).

📊 Model Performance

On test data, the model achieved:

RMSE: 0.0048

MAE: 0.0036

R²: 0.9923

MAPE: 4.84%

✅ This means the model explains 99.2% of the variability in damping ratio and has high predictive accuracy.


📂 Project Structure
│── concrete_damping_dataset_filled.csv   # Preprocessed dataset
│── damp_model_training.py                # Model training & evaluation
│── damping_model.pkl                     # Saved trained model
│── app.py                                # Streamlit web app
│── requirements.txt                      # Dependencies
│── README.md                             # Project documentation

🔮 Future Improvements

-Add more visualization features inside the app.

-Experiment with other ML models (XGBoost, Neural Networks).

-Deploy online (Streamlit Cloud / Heroku / AWS).

🙌 Acknowledgements

Dataset source: (mention your dataset source or say "self-prepared")

Built with Python, Scikit-learn, Pandas, and Streamlit.