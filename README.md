# Breast-Cancer-Detection
Breast Cancer Detection using RandomForestClassifier 

```markdown
# Breast Cancer Detection

```
This project focuses on the detection of breast cancer using machine learning techniques. It involves data exploration, preprocessing, model training, and evaluation. The goal is to build a classification model that can accurately predict whether a breast tumor is malignant (M) or benign (B) based on various features.




import numpy as np
import pandas as pd

```# Load the dataset
data = pd.read_csv('/kaggle/input/breast-cancer-dataset/breast-cancer.csv')
df = pd.DataFrame(data)
## Setting up the Environment
```

## Exploratory Data Analysis

Explore the dataset to understand its structure and characteristics.

```python
# Display dataset information
data.info()

# Display the first few rows of the dataset
data.head()

# Visualize the distribution of the 'diagnosis' column
data["diagnosis"].value_counts()
```

## Handling Categorical Values

Use ordinal encoding to convert categorical values into numerical format.

```python
from sklearn.preprocessing import OrdinalEncoder

# Encode the 'diagnosis' column
ordinal_encoder = OrdinalEncoder()
data_cat = data[['diagnosis']]
data_cat_encoded = ordinal_encoder.fit_transform(data_cat)
```

## Training the Model

Train multiple machine learning models to predict breast cancer diagnosis.

```python
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train different classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Evaluate model accuracy
```


Analyze the model's performance using classification metrics and visualizations.

```python
from sklearn import metrics

# Print classification report
print(metrics.classification_report(Y_test, Y_pred))

# Display confusion matrix as a heatmap
import seaborn as sns
import matplotlib.pyplot as plt

cm = metrics.confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm, annot=True, fmt='.2f')
plt.show()
```

This project provides a comprehensive example of breast cancer detection, from data analysis to model training and evaluation. The classification model's high accuracy and informative error analysis make it a valuable tool for medical diagnosis.


**Contributors**:
- [<img src="https://avatars.githubusercontent.com/u/91361858?v=4" width="40" height="40" alt="GitHub Icon">](https://github.com/SrSurajithPranav)
