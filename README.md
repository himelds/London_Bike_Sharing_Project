# 🚴‍♂️ London Bike-Sharing Demand Prediction using Machine Learning

This repository contains an individual machine learning project focused on predicting demand for London's Santander bike-sharing system using real-world environmental and temporal data. The goal is to build regression models to estimate hourly bike demand and identify the key features influencing usage patterns.

---

## 📌 Project Overview

Bike-sharing systems offer a sustainable and efficient transportation alternative in modern cities. Santander Cycles, London’s bike-sharing scheme, has become an integral part of the city’s transport network. However, operators still face challenges in predicting demand and allocating bikes efficiently. This project aims to apply machine learning to forecast hourly bike demand based on weather and time-related factors, ultimately helping improve operational decisions and service quality.

---

## 🎯 Project Goal

- Analyze factors affecting hourly demand for bike rentals in London
- Build predictive models to forecast bike usage
- Identify the most influential features affecting demand
- Recommend the most effective model for deployment
- Derive actionable insights for improving bike-sharing operations

---

## 📊 Dataset

**Source**: [Kaggle - London Bike Sharing Dataset](https://www.kaggle.com/datasets/hmavrodiev/london-bike-sharing-dataset)

- **Records**: 17,414 hourly entries
- **Features**: Timestamp, Temperature, Humidity, Wind Speed, Weather Code, Holiday/Weekend Flags, Season, etc.
- **Target**: `cnt` – total bike rentals per hour

---

## 🧹 Data Preprocessing

- Dropped correlated feature: `t2` (feels-like temperature)
- Extracted hour, day, month, year from timestamp
- One-hot encoded categorical variables
- Explored outliers using Tukey’s method; retained them after validation
- Verified no missing values

---

## 📈 Exploratory Data Analysis (EDA)

- Higher bike demand during warm and clear weather
- Usage spikes during daytime and working hours
- Seasonality and weekday/weekend trends observed
- Weather and temperature are strongly correlated with demand
- Feature engineering enhanced model insights

---

## 🤖 Machine Learning Models

The following regression models were trained and evaluated:

| Model              | R² Score | RMSE    | MAE    |
|-------------------|----------|---------|--------|
| Linear Regression  | 0.727    | 571.70  | 394.16 |
| Lasso Regression   | 0.727    | 571.59  | 393.81 |
| Random Forest      | 0.940    | 267.99  | 149.63 |
| Gradient Boosting  | 0.944    | 258.44  | 160.78 |
| AdaBoost           | 0.455    | 808.20  | 693.98 |

📌 **Best Model**: Gradient Boosting  
💡 **Recommended Model for Deployment**: Random Forest (robust, scalable, interpretable)

---

## 🛠️ Hyperparameter Tuning

- **AdaBoost** tuning with GridSearchCV
  - Best parameters: `n_estimators=50`, `learning_rate=0.1`
  - Improved R² to 0.5477, still underperformed compared to others
- No further tuning applied to Gradient Boost or Random Forest to avoid overfitting

---

## 🔍 Key Findings

- **Temperature** is the most influential factor for demand, with demand peaking during warm but not extreme temperatures.
- **Clear and dry weather** leads to higher bike usage; rainy or stormy conditions suppress it.
- **Time-based features** (hour, month, season) significantly impact usage trends.
- **Random Forest and Gradient Boosting** yielded the best predictive performance, with R² scores above 0.94.
- Outliers in bike demand were valid (e.g., spikes in summer), so retained.

---

## 🧠 Conclusion

This project demonstrates that machine learning techniques can effectively predict bike-sharing demand using environmental and temporal data. The Gradient Boosting model offered the highest accuracy, while Random Forest is favored for practical deployment due to its flexibility and ease of integration.

Insights from this project can support operational decisions such as:
- Predictive allocation of bikes
- Infrastructure planning (e.g., high-demand zones)
- Resource optimization during peak demand periods

To build a more robust system, additional data like station-level availability, public event schedules, or traffic congestion levels could be integrated in future work.

---

## 🧩 Future Recommendations

- Collect real-time station-level data to develop spatial rebalancing models
- Implement a simple classification system (e.g., low, medium, high demand signals)
- Extend the model with post-pandemic behavior data and public transport disruptions
- Integrate the predictive model into a live dashboard for operations teams

---

## 👨‍💻 Author

**Himel Das**  
*Aspiring Data Scientist | Passionate about ML, analytics, and solving real-world problems*
🔗 [LinkedIn Profile](https://www.linkedin.com/in/dashimel/)
---

## 🔗 Dataset

[https://www.kaggle.com/datasets/hmavrodiev/london-bike-sharing-dataset](https://www.kaggle.com/datasets/hmavrodiev/london-bike-sharing-dataset)

---

## 📄 License

This project is for educational and portfolio purposes only. Not intended for commercial use.

