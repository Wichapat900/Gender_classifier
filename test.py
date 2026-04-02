import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load model
with open('gender_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Test on unseen data
test_df = pd.read_csv('Gender_test.csv')
X_test = test_df[['skirt', 'hair', 'frequency']]
y_test = test_df['gender']

y_pred_test = model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test) * 100:.1f}%")
print(classification_report(y_test, y_pred_test, target_names=['Boy', 'Girl']))