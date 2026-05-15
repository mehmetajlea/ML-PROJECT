# E-Commerce Product Recommendation Predictor 

This project presents a Machine Learning pipeline designed to predict whether a customer will recommend a product based on their review characteristics. It features a complete backend for data processing, model optimization, and a web-based dashboard for real-time predictions.

## Project Overview
In the competitive world of E-commerce, understanding customer feedback is crucial. This project analyzes customer reviews using advanced ML techniques to classify recommendation intent. It effectively handles imbalanced datasets and provides an interpretable model for business decision-making.



[Image of machine learning pipeline for classification]


##  Advanced Tech Stack
* **Web Interface:** Streamlit (Interactive Dashboard)
* **Optimization:** Optuna (Bayesian Hyperparameter Tuning)
* **Model Explainability:** SHAP (Shapley Additive Explanations)
* **Data Processing:** Scikit-Learn Pipelines & ColumnTransformers
* **Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique)
* **Model:** Logistic Regression & SGDClassifier
* **Serialization:** Joblib

##  Technical Workflow
1. **Data Preprocessing:** Handling missing values via `KNNImputer` and encoding categorical variables using `OneHotEncoder`.
2. **Feature Engineering:** Applying `PCA` (Principal Component Analysis) for dimensionality reduction and `StandardScaler` for normalization.
3. **Imbalance Correction:** Integrated `SMOTE` within a specialized `ImbPipeline` to ensure fair training and prevent data leakage.
4. **Hyperparameter Tuning:** Automated search for the best parameters using **Optuna** to maximize the ROC-AUC score.
5. **Interpretability:** Utilizing **SHAP** values to visualize which features (like Rating or Positive Feedback Count) drive the model's decisions.



## Features of the Web App
* **Real-time Prediction:** Input customer review features to see if a product will be recommended.
* **Data Export:** Capability to download prediction results as a **CSV file** for further analysis.
* **Visual Insights:** Dynamic dashboard displaying model performance and data distributions.

## Installation & Usage

1. **Clone the repository:**
```bash
git clone [https://github.com/mehmetajlea/ML-PROJECT.git](https://github.com/mehmetajlea/ML-PROJECT.git)
cd ML-PROJECT
```
2.Install dependencies:
```bash
pip install -r requirements.txt
```
3.Launch the Web App:
```bash
streamlit run app.py
```

## Credits

Author: **Lea Mehmetaj**

Mentor: **Dr. Sc. Liridon Hoti**

Institution: **UBT - Faculty of Mechatronics Engineering**

Course: **Design Data Processing Systems / Machine Learning**

## License

This project is intended for educational and research purposes as part of the Mechatronics Engineering curriculum at UBT.
