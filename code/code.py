import warnings
warnings.filterwarnings('ignore') 
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             RocCurveDisplay)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def main():
    # Load Dataset & Info
    file_path = r"C:\Users\HP\OneDrive\Desktop\ML-PROJECT\data\Womens Clothing E-Commerce Reviews.csv"

    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # Data Cleaning & Feature Engineering
    cols_to_drop = ['Unnamed: 0', 'Clothing ID']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    if 'Review Text' in df.columns:
        df['Review Text'] = df['Review Text'].fillna('')
        df['Text Length'] = df['Review Text'].astype(str).apply(len)
    else:
        df['Review Text'] = "" 
        df['Text Length'] = 0

    if 'Title' in df.columns:
        df = df.drop(columns=['Title'])

    target = "Recommended IND"
    if target not in df.columns:
        raise ValueError(f"Target '{target}' missing!")

    df = df.dropna(subset=[target])

    # EDA
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target in numerical_cols: numerical_cols.remove(target)

    if len(numerical_cols) > 0:
        df[numerical_cols].hist(figsize=(12, 8), bins=20, edgecolor='black')
        plt.suptitle("Histograms of Numerical Features")
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

    # Split Data
    X = df.drop(columns=[target, 'Review Text']) 
    y = df[target]
    reviews_text = df['Review Text']

    X_train_full, X_temp, y_train_full, y_temp, reviews_train_full, reviews_temp = train_test_split(
        X, y, reviews_text, test_size=0.3, random_state=42, stratify=y)

    X_val, X_test, y_val, y_test, reviews_val, reviews_test = train_test_split(
        X_temp, y_temp, reviews_temp, test_size=0.5, random_state=42, stratify=y_temp)

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

    # Automated Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', KNNImputer(n_neighbors=5)),  # Imputation
                ('scaler', StandardScaler())             # Scaling
            ]), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])

    print("\nPreview of Processed Data (Normalized & Fixed)")
    X_train_processed_array = preprocessor.fit_transform(X_train_full)
    ohe = preprocessor.named_transformers_['cat']
    ohe_feature_names = ohe.get_feature_names_out(categorical_cols)
    all_feature_names = numerical_cols + list(ohe_feature_names)
    processed_df = pd.DataFrame(X_train_processed_array, columns=all_feature_names)
    print(processed_df.head().round(2))
    print("(Note: Numerical values are now normalized. Categories are OneHotEncoded.)")

    # Model Training
    print("\nTraining Model: Logistic Regression")

    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('pca', PCA(n_components=0.95)),      
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])

    param_grid = {
        'pca__n_components': [0.90, 0.95],
        'classifier__C': [0.1, 1, 10],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs']
    }

    cv = StratifiedKFold(n_splits=5)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
    grid.fit(X_train_full, y_train_full)

    print("Best Parameters:", grid.best_params_)
    best_model = grid.best_estimator_

    # Evaluation Function
    def evaluate(model, X, y, name):
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        print(f"\n--- {name} Metrics ---")
        print(f"Accuracy:  {accuracy_score(y, y_pred):.3f}")
        print(f"Precision: {precision_score(y, y_pred):.3f}")
        print(f"Recall:    {recall_score(y, y_pred):.3f}")
        print(f"F1-score:  {f1_score(y, y_pred):.3f}")
        print(f"AUC-ROC:   {roc_auc_score(y, y_prob):.3f}")

        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix ({name})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        RocCurveDisplay.from_estimator(model, X, y)
        plt.title(f"ROC Curve ({name})")
        plt.show()
        return y_pred, y_prob

    y_pred_test, y_prob_test = evaluate(best_model, X_test, y_test, "Test Set")

    # PCA Visualization
    print("\nPCA Visualization (Test Set)")
    X_test_preprocessed = best_model.named_steps['preprocessor'].transform(X_test)
    pca_obj = best_model.named_steps['pca']
    X_test_pca = pca_obj.transform(X_test_preprocessed)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_test_pca[:, 0], y=X_test_pca[:, 1], hue=y_test, palette='viridis', alpha=0.6)
    plt.title("PCA: 2D Visualization of Test Set Classes")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Recommendation")
    plt.show()

    # Client Review Examples
    print("\nClient Review Examples (Output)")
    results_df = pd.DataFrame({
        'Client Review': reviews_test.values,
        'Actual Label': y_test.values,
        'Predicted Label': y_pred_test,
        'Confidence (%)': (y_prob_test * 100).round(1)
    })

    label_map = {0: "Not Recommended", 1: "Recommended"}
    results_df['Actual Label'] = results_df['Actual Label'].map(label_map)
    results_df['Predicted Label'] = results_df['Predicted Label'].map(label_map)

    sampled_results = results_df.sample(5, random_state=42)
    pd.set_option('display.max_colwidth', 100) 
    pd.set_option('display.colheader_justify', 'center')
    print(sampled_results[['Client Review', 'Actual Label', 'Predicted Label', 'Confidence (%)']])

    # Feature Importance
    print("\nFeature Importance Analysis")
    pipe_no_pca = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])
    pipe_no_pca.fit(X_train_full, y_train_full)

    ohe = pipe_no_pca.named_steps['preprocessor'].transformers_[1][1]
    ohe_names = ohe.get_feature_names_out(categorical_cols)
    all_feature_names = np.concatenate([numerical_cols, ohe_names])

    coeffs = pipe_no_pca.named_steps['classifier'].coef_[0]
    importance_df = pd.DataFrame({'feature': all_feature_names, 'coef': coeffs})
    importance_df['abs_coef'] = importance_df['coef'].abs()
    importance_df = importance_df.sort_values('abs_coef', ascending=False).head(15)

    plt.figure(figsize=(10,6))
    sns.barplot(x='abs_coef', y='feature', data=importance_df, hue='feature', palette='viridis', legend=False)
    plt.title("Top 15 Factors Influencing Prediction")
    plt.show()

    # Automated Prediction on New Data
    print("\n Automated Prediction on New Data")
    new_client_data = pd.DataFrame({
        'Age': [45],
        'Rating': [5],
        'Positive Feedback Count': [10],
        'Division Name': ['General'],
        'Department Name': ['Dresses'],
        'Class Name': ['Dresses'],
        'Text Length': [150]
    })

    print("\nRaw New Data (Unprocessed):")
    print(new_client_data.T) # Transpose to display vertically

    prediction = best_model.predict(new_client_data)
    probability = best_model.predict_proba(new_client_data)[0][1]

    print(f"\nPipeline Output:")
    print(f"Prediction: {prediction[0]} (0=Not Recommended, 1=Recommended)")
    print(f"Confidence:  {probability*100:.2f}%")
    print("\nNote: The pipeline automatically handled Imputation, Scaling, Encoding, and Feature Selection for this new data.")

if __name__ == "__main__":
    main()
