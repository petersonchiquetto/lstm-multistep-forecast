# 🌦️ LSTM Weather Forecast

This project implements a **multi-step time series forecasting model** using an LSTM neural network in TensorFlow/Keras. It simulates weather conditions (humidity, pressure, wind) and predicts **temperature for the next 3 days** based on the past 7 days of data.

---

## 📌 Purpose

The goal of this project is to demonstrate how a deep learning model can forecast future values in a time series — specifically temperature — using:

- Synthetic multivariate weather data
- LSTM layers for temporal learning
- Multi-output prediction for the next 3 days

---

## 📊 Features

- Synthetic daily weather dataset generator (365 days)
- Normalization using `MinMaxScaler`
- Sliding windows to create sequences for supervised learning
- LSTM model with `Dropout` and `Dense` layers
- Multi-output regression (3 future time steps)
- Inverse normalization of predictions
- Plot of real vs predicted temperatures for each day

---

## 🧠 Technologies Used

| Library         | Purpose                                 |
|----------------|------------------------------------------|
| `NumPy`         | Data generation                         |
| `Pandas`        | Data manipulation                       |
| `Matplotlib`    | Plotting results                        |
| `scikit-learn`  | Data normalization (`MinMaxScaler`)     |
| `TensorFlow`    | Building and training the LSTM model    |

---

## 🚀 How It Works

1. **Generate Synthetic Data**  
   Simulates daily values for humidity, pressure, wind, and temperature.

2. **Normalize and Create Sequences**  
   Uses 7 days of input to predict the temperature for the next 3 days.

3. **Build and Train Model**  
   - LSTM(64) → Dropout(0.2) → LSTM(32) → Dense → Output (3 values)

4. **Evaluate and Predict**  
   Evaluates on test set using MAE, and plots predictions vs real values.

---

## 📈 Sample Output

```text
Erro médio absoluto no teste: 1.32 °C
````

---

## 📦 Installation

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

---

## 📝 License

This project is licensed under the MIT License.

---

## 👤 Author

Developed by [Peterson Chiquetto](https://github.com/petersonchiquetto)

---

## 💡 Future Improvements

* Load real weather datasets (e.g., from OpenWeatherMap or NOAA)
* Add temperature trend detection
* Experiment with CNN-LSTM or Transformer models
* Export model for deployment (e.g., TensorFlow Lite)
