import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Step 1 : Load dataset (Titanic)
df = pd.read_csv('tested.csv')
print("Original Dataset :")
print(df.head())

# Step 2 : Explore data
# Gives the info of datatypes and more
print("\nDataset Info:")
print(df.info())

# Description of dataset
print("\nDescription of dataset :")
print(df.describe())

# Count of total null values present in each cloumn
print("\nMissing Values:")
print(df.isnull().sum())

# Step 3 : Handle missing data
# Drop column with most null values i.e. Cabin
df.drop(columns = ['Cabin'], inplace = True)

# Impute numerical column with mean
imputer = SimpleImputer(strategy = 'mean')
df['Age'] = imputer.fit_transform(df[['Age']])
df['Fare'] = imputer.fit_transform(df[['Fare']])


# Impute categorical column with mode
imputer = SimpleImputer(strategy = 'most_frequent')
df['Embarked'] = imputer.fit_transform(df[['Embarked']]).ravel()

# Drop missing rows
df.dropna(inplace = True)

print("\nAfter Handling Missing Data:")
print(df.isnull().sum())

# Step 4 : Handle Categorical Variables
# Identify categorical columns
categorical_cols = df.select_dtypes(include = 'object').columns.tolist()
print("\nCategorical Columns:", categorical_cols)

# Apply LabelEncoder
# The 'class' column is not in the dataframe. Let's remove this part.
# class_order = {'First' : 1, 'Second' : 2, 'Third' : 3}
# df['class'] = df['class'].map(class_order)

# Apply OneHOtEncoder
# Based on the output of `categorical_cols`, the categorical columns are 'Name', 'Sex', and 'Ticket'.
# 'Name' and 'Ticket' are not suitable for one-hot encoding. Let's encode 'Sex' and 'Embarked' instead.
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
print("\nAfter Encoding Categorical Variables:")
print(df.head())

# Step 5 : Apply Normalization and Standardization
# Identify numerical columns
numerical_cols = df.select_dtypes(include = ['int64', 'float64']).columns.tolist()
print("\nNumerical Columns:", numerical_cols)

scaler = StandardScaler();
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("\nAfter Scaling (StandardScaler):")
print(df[numerical_cols].head())

# Step 6 : Split the dataset
# Define features and target
X = df.drop(columns='Survived')
Y = df['Survived']

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("\nShapes of Train/Test Sets:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", Y_train.shape)
print("y_test:", Y_test.shape)
