import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Örnek veri
df = pd.read_csv("datasets/cleaned_data_v2.csv")
target_variable = 'SubscribedToTermDeposit'
X = df.drop(columns=[target_variable])
y = df[target_variable]

# Categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['number']).columns

# Create pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combine into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

# Fit the preprocessor
preprocessor.fit(X)

# Transform the data
X_processed = preprocessor.transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Train Logistic Regression model
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(X_train, y_train)

# Save the preprocessor
with open("preprocessor2.pkl", "wb") as preprocessor_file:
    pickle.dump(preprocessor, preprocessor_file)

# Save Random Forest model
with open("best_rf2_model.pkl", "wb") as rf_file:
    pickle.dump(rf_model, rf_file)

# Save KNN model
with open("best_knn2_model.pkl", "wb") as knn_file:
    pickle.dump(knn_model, knn_file)

# Save Logistic Regression model
with open("best_logreg2_model.pkl", "wb") as logreg_file:
    pickle.dump(logreg_model, logreg_file)

print("Preprocessor and all models saved successfully!")

