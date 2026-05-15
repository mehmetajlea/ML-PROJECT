import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import optuna
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
optuna.logging.set_verbosity(optuna.logging.WARNING)

#LOAD & PREPARE DATA (Same as standard training)
file_path = r"C:\Users\HP\OneDrive\Desktop\ML-PROJECT\data\Womens Clothing E-Commerce Reviews.csv"
df = pd.read_csv(file_path)

cols_to_drop = ['Unnamed: 0', 'Clothing ID', 'Title']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
df['Review Text'] = df['Review Text'].fillna('')
df['Text Length'] = df['Review Text'].astype(str).apply(len)
df = df.drop(columns=['Review Text'])
df = df.dropna(subset=['Recommended IND'])

X = df.drop(columns=['Recommended IND'])
y = df['Recommended IND']
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

# Preprocess data once (Needed for SHAP and Sensitivity)
X_train_proc = preprocessor.fit_transform(X_train_full)
X_test_proc = preprocessor.transform(X_test)

#OPTUNA (Smart Hyperparameter Optimization)
print("\n- Running Optuna Hyperparameter Optimization -")

def objective(trial):
    param_grid = {
        'classifier__C': trial.suggest_float('C', 0.01, 10.0, log=True),
        'classifier__solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear']),
        'pca__n_components': trial.suggest_float('pca', 0.85, 0.99)
    }
    
    opt_pipe = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('pca', PCA()),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])
    
    cv = StratifiedKFold(n_splits=3)
    score = cross_val_score(opt_pipe, X_train_full, y_train_full, cv=cv, scoring='f1', n_jobs=-1).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10) # Keep low (10) for speed, increase for better results

print("Best Params found by Optuna:", study.best_params)
print("Best F1 Score:", study.best_value)

# SHAP VALUES (Fixed)
print("\n- Computing SHAP Values -")

# Train a simple Logistic Regression (without PCA) for SHAP interpretation
lr_shap = LogisticRegression(class_weight='balanced', max_iter=1000)
lr_shap.fit(X_train_proc, y_train_full)

# Create Explainer
explainer = shap.LinearExplainer(lr_shap, X_train_proc)

# Calculate SHAP values for a sample
X_sample = X_test_proc[:50] # First 50 rows
shap_values = explainer.shap_values(X_sample)

# Define Feature Names (Required for the plot) 
ohe = preprocessor.named_transformers_['cat']
ohe_names = ohe.get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols + list(ohe_names)

#Handle shape mismatch 
if len(shap_values.shape) == 3:
    shap_values = shap_values[:, 1, :] 

plt.title("SHAP Summary Plot (Local Explanations)")
shap.summary_plot(shap_values, X_sample, feature_names=all_feature_names)
plt.show()

# SENSITIVITY ANALYSIS (Fixed for Mixed Data Types)
print("\nPerforming Sensitivity Analysis (Age Impact)")

baseline_df = X_test.iloc[[0]].copy() 

# Range of ages to test
ages = range(18, 70, 5)
probs = []

print("Calculating probability for different ages...")

#Vary Age, Keep everything else constant
for age in ages:
    temp_df = baseline_df.copy()
    temp_df['Age'] = age
    
    # Preprocess the data (Same pipeline used for SHAP)
    temp_proc = preprocessor.transform(temp_df)
    
    # Predict Probability (Class 1 = Recommended)
    # We use 'lr_shap' model which is already trained
    prob = lr_shap.predict_proba(temp_proc)[0][1]
    probs.append(prob)

plt.figure(figsize=(8, 5))
plt.plot(ages, probs, marker='o', color='blue', linestyle='-')
plt.title("Sensitivity Analysis: Impact of Age on Recommendation")
plt.xlabel("Client Age")
plt.ylabel("Probability of Recommendation")
plt.grid(True)
plt.show()

#INCREMENTAL LEARNING (Streaming Simulation)
print("\n- Simulating Incremental Learning (Streaming Data) -")

# SGDClassifier supports partial_fit
inc_model = SGDClassifier(loss='log_loss', warm_start=True, random_state=42)

# Simulate chunks of data
chunk_size = 1000
accuracies = []

print(f"Training incrementally with chunk size: {chunk_size}")

for i in range(0, len(X_train_proc), chunk_size):
    X_chunk = X_train_proc[i:i+chunk_size]
    y_chunk = y_train_full[i:i+chunk_size].values
    
    #Update model
    inc_model.partial_fit(X_chunk, y_chunk, classes=[0, 1])
    
    # Check accuracy on current chunk
    score = inc_model.score(X_chunk, y_chunk)
    accuracies.append(score)
    
    if (i // chunk_size) % 5 == 0: # Print every 5 chunks
        print(f"Chunk {i/chunk_size}: Accuracy {score:.3f}")

# Plot Learning Curve
plt.figure(figsize=(8, 5))
plt.plot(range(len(accuracies)), accuracies, color='green')
plt.title("Incremental Learning Progress (Online)")
plt.xlabel("Chunk Number")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
print("\n- Advanced Analysis Complete -")