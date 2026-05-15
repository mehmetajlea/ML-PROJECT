import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

print("Loading dataset...")

# Load Data (Using relative path ../data/ because we are inside the /app folder)
file_path = r"../data/Womens Clothing E-Commerce Reviews.csv"
df = pd.read_csv(file_path)

# Cleaning
cols_to_drop = ['Unnamed: 0', 'Clothing ID', 'Title']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
df['Review Text'] = df['Review Text'].fillna('')
df['Text Length'] = df['Review Text'].astype(str).apply(len)
df = df.drop(columns=['Review Text'])
df = df.dropna(subset=['Recommended IND'])

# Split
X = df.drop(columns=['Recommended IND'])
y = df['Recommended IND']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Preprocess
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

# Train Pipeline (Fast version)
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('pca', PCA(n_components=0.95)),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
])

param_grid = {'classifier__C': [0.1]} # Using 0.1 as best param found earlier

print("Training model...")
grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

final_model = grid.best_estimator_

# Save Model
joblib.dump(final_model, 'rf_model.pkl')
print("Model saved as rf_model.pkl")

# Calculate and Save Feature Importance
pipe_no_pca = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
])
pipe_no_pca.fit(X_train, y_train)
ohe = pipe_no_pca.named_steps['preprocessor'].transformers_[1][1]
ohe_names = ohe.get_feature_names_out(categorical_cols)
all_feature_names = np.concatenate([numerical_cols, ohe_names])
coeffs = pipe_no_pca.named_steps['classifier'].coef_[0]
importance_df = pd.DataFrame({'feature': all_feature_names, 'coef': coeffs})
importance_df['abs_coef'] = importance_df['coef'].abs()
importance_df = importance_df.sort_values('abs_coef', ascending=False).head(10)

importance_df.to_csv('feature_importance.csv', index=False)
print("Feature importance saved as feature_importance.csv")