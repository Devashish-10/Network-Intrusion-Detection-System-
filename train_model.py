import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Paths to the training and testing datasets
training_file_path = 'Dataset/UNSW_NB15_Train-set.csv'
testing_file_path = 'Dataset/UNSW_NB15_Test-set.csv'

# Load the datasets
train_df = pd.read_csv(training_file_path)
test_df = pd.read_csv(testing_file_path)

# Drop the 'Unnamed: 0' and 'id' columns if they exist
train_df = train_df.drop(columns=['Unnamed: 0'], errors='ignore')
test_df = test_df.drop(columns=['id'], errors='ignore')

# Encode categorical variables
categorical_columns = ['proto', 'service', 'state', 'attack_cat']
for col in categorical_columns:
    train_df[col] = train_df[col].astype('category').cat.codes
    test_df[col] = test_df[col].astype('category').cat.codes

# Feature extraction using correlation
corr_matrix = train_df.corr().abs()
high_corr_features = corr_matrix.index[corr_matrix['label'] >= 0.25].tolist()
if 'label' not in high_corr_features:
    high_corr_features.append('label')

# Filter the datasets based on high correlation features
train_df = train_df[high_corr_features]
test_df = test_df[high_corr_features]

# Normalize numerical columns
numerical_columns = [col for col in high_corr_features if col not in categorical_columns and col != 'label']
scaler = StandardScaler()
train_df[numerical_columns] = scaler.fit_transform(train_df[numerical_columns])
test_df[numerical_columns] = scaler.transform(test_df[numerical_columns])

# Separate features and labels
X_train = train_df.drop('label', axis=1)
y_train = train_df['label']
X_test = test_df.drop('label', axis=1)
y_test = test_df['label']

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # retain 95% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Define a function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    print("Classification Report for training data:")
    print(classification_report(y_train, y_pred_train))
    print("Classification Report for testing data:")
    print(classification_report(y_test, y_pred_test))
    
    print("Confusion Matrix for training data:")
    print(confusion_matrix(y_train, y_pred_train))
    print("Confusion Matrix for testing data:")
    print(confusion_matrix(y_test, y_pred_test))
    
    print("ROC-AUC Score for testing data:", roc_auc_score(y_test, y_pred_test))
    print("F1 Score for testing data:", f1_score(y_test, y_pred_test))

# SVM with different kernels
kernels = ['linear', 'rbf', 'poly']
for kernel in kernels:
    print(f"\nTraining SVM with {kernel} kernel...")
    svm_model = SVC(kernel=kernel)
    svm_model.fit(X_train_pca, y_train)
    evaluate_model(svm_model, X_train_pca, y_train, X_test_pca, y_test)

# Hyperparameter tuning for SVM with RBF kernel
param_grid = {
    'C': [0.1, 1],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train_pca, y_train)

print("\nBest parameters found by GridSearchCV:")
print(grid.best_params_)

# Evaluate the best model
best_svm_model = grid.best_estimator_
evaluate_model(best_svm_model, X_train_pca, y_train, X_test_pca, y_test)

# Save the best model
model_directory = 'Model'
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
model_path = os.path.join(model_directory, 'svm_model_kernel.pkl')
with open(model_path, 'wb') as file:
    pickle.dump(best_svm_model, file)

# Visualize correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.show()

# Visualize PCA explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(pca.explained_variance_ratio_.cumsum(), marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.show()
