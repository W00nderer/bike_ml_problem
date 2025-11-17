import pandas as pd  # Dataframe
from sklearn.preprocessing import StandardScaler # Scaling skewed numerical values
from sklearn.compose import ColumnTransformer # Transforming columns
from sklearn.pipeline import Pipeline # Pipeline
from xgboost import XGBRegressor # XGBoost 
from category_encoders.target_encoder import TargetEncoder # Encosing categorical cols
from skopt import BayesSearchCV # Bayes Search for parameter hypertuning
from skopt.space import Real, Integer # For parameter range


# Breaking down the Date column into more useful information like year, month, day of the week, etc.
def date_breakdown(df):
    df["Year"] = df["Date"].apply(lambda x: int(x.split("/")[2]))
    df["Month"] = df["Date"].apply(lambda x: int(x.split("/")[1]))
    df["Day"] = df["Date"].apply(lambda x: int(x.split("/")[0]))

    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    return df.drop(columns=['Date']) # drop the Date column afterwards

# New engineering features of columns with medium-high correlation
def correlation(df):
    df = df.copy() # working with a copy of the DataFrame to avoid modifying the original
    correlation_pairs = [['Temperature(°C)','Dew point temperature(°C)'],
                        ["Humidity(%)", 'Visibility (10m)'],
                        ["Humidity(%)",'Solar Radiation (MJ/m2)']]
    for col1, col2 in correlation_pairs:
        df[f'{col1[:3]}_minus_{col2[:3]}'] = df[col1] - df[col2] # difference
        df[f'{col1[:3]}_div_{col2[:3]}'] = df[col1] / (df[col2] + 1e-6) # ratio
        df[f'{col1[:3]}_avg_{col2[:3]}'] = (df[col1] + df[col2]) / 2 #average
    return df.drop(columns=['Temperature(°C)','Dew point temperature(°C)']) # Drop the highly correlated columns

# Finding the average of categorical columns with high importance value           
def average(df_train, df_test):
    df_train = df_train.copy()
    df_test = df_test.copy()
    average_cols = ['Holiday','IsWeekend','Seasons','Month']
    for col in average_cols:
        avg = df_train.groupby(col)['Rented Bike Count'].mean().to_dict() # Rented bike mean of selected column
        df_train[f'Avg{col}'] = df_train[col].map(avg) 
        df_test[f'Avg{col}'] = df_test[col].map(avg) 
    return df_train.drop(columns=['Rented Bike Count']), df_test # Drop the Y column in train afterwards

# Get the X_train, Y_train from train dataset
X_train = pd.read_csv("train.csv")
Y_train = X_train['Rented Bike Count']
# Preprocessing of X_train
X_train = date_breakdown(X_train)
X_train = correlation(X_train)

# Get the X_test, Y_test
X_test = pd.read_csv("test.csv")
Y_test = pd.read_csv('sample_submission.csv')
# Preprocessing of X_test
X_test = date_breakdown(X_test)
X_test = correlation(X_test)

#More preprocessing
X_train, X_test = average(X_train,X_test)

# Numeric columns list for scaling
numeric_cols = list(X_train.select_dtypes(include='number').columns)
numeric_cols = [col for col in numeric_cols if col != 'ID']

# Categorical columns to encode
to_encode = ["Seasons","Holiday","Functioning Day"]

# Preproecssor setup
preprocessor = ColumnTransformer(
    transformers=[
        ('target_enc', TargetEncoder(), to_encode),
        ('num', StandardScaler(), numeric_cols)
    ],

    remainder='passthrough'
)
# Estimators setup
estimators = [
    ('preprocessor', preprocessor),
    ('clf', XGBRegressor(random_state=8,
                         booster='gbtree',
                         n_estimators=1000))
]

#Pipeline setup
pipe = Pipeline(steps=estimators)

# Hyperparametes 
search_space = {
    'clf__max_depth': Integer(3,15),
    'clf__learning_rate': Real(0.0001, 0.3, prior='log-uniform'),
    'clf__subsample': Real(0.1, 1.0),
    'clf__colsample_bytree': Real(0.1, 1.0),
    'clf__colsample_bylevel': Real(0.1, 1.0),
    'clf__colsample_bynode': Real(0.1, 1.0),
    'clf__reg_alpha': Real(1.0, 7.0),
    'clf__reg_lambda': Real(1.0, 7.0),
    'clf__gamma': Real(0.0, 10.0)
}

# Searching for the optimal model
opt = BayesSearchCV(pipe, search_space, cv=5, n_iter=100,
                    scoring='neg_root_mean_squared_error', random_state=8)

# FItting
opt.fit(X_train, Y_train)

# Saving the prediction
pred = opt.predict(X_test)
submission = pd.DataFrame({
    'ID': X_test['ID'].astype(int),
    'Rented Bike Count': pred
})
submission.to_csv('my_submission.csv', index=False)