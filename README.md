#  Task 1: Data Cleaning and Preprocessing
 ##  Objective
 To learn and apply fundamental techniques for cleaning and preparing raw data for
 machine learning using Python.--
##  Tools & Libraries Used- **Python**- **Pandas**  Data manipulation- **NumPy**  Numerical operations- **Matplotlib** & **Seaborn**  Visualization- **Scikit-learn**  Preprocessing utilities (like StandardScaler)--
##  Steps Performed
 1. **Dataset Loading and Exploration**
   - Loaded the dataset using `pandas`
   - Explored data using `.info()`, `.describe()`, and `.isnull().sum()`
 2. **Handling Missing Values**
   - Filled missing numerical values (like `Age`) using median
   - Imputed categorical missing values (like `Embarked`) using mode
   - Dropped features with excessive missing values (e.g., `Cabin`)
 3. **Categorical Encoding**
   - Converted categorical variables (`Sex`, `Embarked`) into numeric format using
 `pd.get_dummies()`
 4. **Feature Scaling**
   - Standardized numerical columns (`Age`, `Fare`, `SibSp`, `Parch`) using
 `StandardScaler`
 5. **Outlier Detection & Removal**
   - Visualized numerical columns with boxplots
   - Removed outliers using Z-score method
 6. **Final Cleaned Dataset**
   - Exported the cleaned dataset for use in modeling tasks--
##  Output
 A clean, preprocessed dataset ready for machine learning models.--
##  Learnings- Identifying and treating missing values- Encoding categorical data- Normalizing data for consistent scale
- Visualizing and removing outliers- End-to-end preprocessing pipeline for real-world data--
##  Code Snippets
 ###  Load and Explore the Dataset
 ```python
 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 import seaborn as sns
 df = pd.read_csv("titanic.csv")
 print(df.info())
 print(df.describe())
 print(df.isnull().sum())
 ```
 ###  Handle Missing Values
 ```python
 df['Age'].fillna(df['Age'].median(), inplace=True)
 df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
 df.drop(columns=['Cabin'], inplace=True)
 ```
 ###  Convert Categorical Features
 ```python
 df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
 df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)
 ```
 ###  Standardize Numerical Features
 ```python
 from sklearn.preprocessing import StandardScaler
 scaler = StandardScaler()
 num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
 df[num_cols] = scaler.fit_transform(df[num_cols])
 ```
 ###  Visualize and Remove Outliers
 ```python
 from scipy import stats
 sns.boxplot(data=df[num_cols])
 plt.show()
 z_scores = np.abs(stats.zscore(df[num_cols]))
 df = df[(z_scores < 3).all(axis=1)]
 ```
