import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#%%
path = r"C:/Users/Saad Raja/Desktop/Data Science Project/Telco-Customer-Churn.csv"
# Load the dataset
df = pd.read_csv(path)

# Exploratory Data Analysis (EDA)
print("Dataset shape:", df.shape)
print("Columns:", df.columns)
print("Data types:\n", df.dtypes)
print("Missing values:\n", df.isnull().sum())

#%%

# Data Cleaning and Preprocessing
df.replace(' ', np.nan, inplace=True)
df.dropna(inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

#%%

# Data Visualization
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()

#%%

# Data Encoding
cat_columns = df.select_dtypes(include='object').columns
le = LabelEncoder()
df[cat_columns] = df[cat_columns].apply(le.fit_transform)

#%%

# Feature Selection and Train-Test Split
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%

# Model Training and Evaluation
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#%%

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
