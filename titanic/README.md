Titanic Survival Prediction
1. Overview
This project tackles the classic Kaggle competition, "Titanic: Machine Learning from Disaster." The goal is to build a predictive model that answers the question: "what sorts of people were more likely to survive?" using passenger data (e.g., name, age, gender, socio-economic class, etc.).

This repository contains a Jupyter Notebook (titanic_ml.ipynb) that walks through the entire machine learning workflow, from initial data exploration to feature engineering, model training, hyperparameter tuning, and final prediction submission.

2. Methodology
The project follows a structured approach to solving this classification problem:

2.1. Exploratory Data Analysis (EDA)
Initial Data Inspection: The train.csv and test.csv datasets were loaded and inspected for data types, shapes, and missing values.

Visualization: Key relationships between features and the survival outcome were visualized using seaborn and matplotlib. EDA revealed that:

Females had a much higher survival rate than males.

Passengers in Pclass 1 had a higher survival rate than those in Pclass 2 or 3.

Children (lower age groups) had a higher chance of survival.

2.2. Feature Engineering and Preprocessing
Several preprocessing steps were taken to clean the data and create more informative features:

Handling Missing Values:

Embarked: The two missing values in the training set were imputed with 'C' (Cherbourg), which was the mode for passengers with a similar Passenger Class and Fare.

Age: Missing Age values were filled using the mean age calculated from the training data. This same mean was applied to the validation and test sets to prevent data leakage.

Fare: A single missing Fare in the test set was imputed using the mode of passengers with a similar Pclass and other related features.

Cabin: The first letter of the Cabin number was extracted to create a new categorical feature representing the deck. Missing values were filled with 'U' for 'Unknown'.

Creating New Features:

Title: Titles (Mr, Mrs, Miss, etc.) were extracted from the Name column. Rare titles were grouped into more common ones (e.g., 'Dr', 'Col' became 'Mr'; 'Lady', 'Countess' became 'Mrs').

AgeGroup: The continuous Age feature was binned into categorical age groups (e.g., 0-12, 13-25).

FamilyGroup: SibSp (siblings/spouses) and Parch (parents/children) were combined to create a FamilySize feature, which was then categorized into groups: 'Alone', 'Small Family' (2-4), and 'Large Family' (4+).

Encoding and Scaling:

Categorical Variables: Features like Sex, Embarked, Title, AgeGroup, and FamilyGroup were converted into numerical format using one-hot encoding (pd.get_dummies).

Feature Scaling: StandardScaler was applied to the data before training models like SVM and KNN, which are sensitive to the scale of input features.

2.3. Modeling
Data Splitting: The training data was split into training and validation sets (80/20 split) using stratification to maintain the same proportion of survival outcomes in both sets.

Model Selection: Several classification models were trained and evaluated:

RandomForestClassifier (primarily used for feature importance analysis)

LightGBM Classifier

GradientBoostingClassifier

AdaBoostClassifier (using sample_weight to handle class imbalance)

Support Vector Machine (SVC)

K-Nearest Neighbors (KNN)

Hyperparameter Tuning: GridSearchCV and RandomizedSearchCV were used to find the optimal hyperparameters for each model, using ROC AUC as the primary scoring metric.

3. Results
The models were evaluated on the hold-out validation set. The Gradient Boosting Classifier was selected for the final submission due to its strong and stable performance.

Best Model (Gradient Boosting):

Validation Accuracy: 79.3%

Validation AUC: 0.828

The model's predictions on the test set were generated and saved to submission.csv.
