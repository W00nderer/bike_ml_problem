# Bike Sharing Demand Problem

I have attempted to solve my first regression machine learning problem using the knowledge I gained in my Machine learning class.  I chose to use the XGBoost model and Bayes Search CV for parameter hypertuning, as well as utilizing different preprocessing techniques. 

## Table of Contents:

  - Preprocessing
  - Hyperparameter tuning
  - Kaggle Score
  - Further Improvements
  
## Preprocessing:

I started by analyzing the datasets provided. The data had no missing values or outliers, but some numeric columns were skewed, so I used StandardScaler() to standardize the features. Furthermore, I encoded the categorical columns using TargetEncoder(). Both of these functions were combined in a single ColumnTransformer():

<img width="681" height="390" alt="image" src="https://github.com/user-attachments/assets/c20bea1f-2dce-4a90-b5e3-f60cc6ac4ec3" />

One of the interesting columns given was ‘Date’. While on its own it couldn’t provide us with much information, I extracted different features like Month, Day of the Week, and others that provided more insight into the data:

<img width="935" height="242" alt="image" src="https://github.com/user-attachments/assets/35ecc5c5-e5aa-47ef-ba11-46daba18209a" />

Next, using the built-in feature of XGBoost, I plotted the feature importance of the columns:

<img width="903" height="533" alt="image" src="https://github.com/user-attachments/assets/0ba108b3-24c9-42f7-ba49-ac8ad5c0a407" />

Looking at the chart, there are a couple of categorical columns with great importance, such as Holiday, Seasons, and newly extracted IsWeekend and Month. I found the average of those columns for more insight:

<img width="945" height="225" alt="image" src="https://github.com/user-attachments/assets/1a7e7e76-e186-4afd-972a-4ecd29899233" />

Moreover, I plotted a heatmap to find correlation between columns:

<img width="898" height="663" alt="image" src="https://github.com/user-attachments/assets/4e01dc5b-e3d9-46a8-9109-d27dbb1bce5b" />

In the heatmap, we can see high positive correlation between Temperature and Dew point temperature columns, as well as a mild negative correlation between Humidity and Visibility, Solar Radiation. Using this knowledge, I found the difference, ratio and average between these columns:

<img width="941" height="238" alt="image" src="https://github.com/user-attachments/assets/9eb53da9-e12a-4de8-972c-e5597730440e" />

In the program, I used Pipeline technique to combine preprocessing and model training into one workflow, ensuring that every preprocessing step is applied consistently to both training and test data:

<img width="556" height="307" alt="image" src="https://github.com/user-attachments/assets/356cf5fd-88b5-42a5-92ad-eadabcf32a0d" />

## Hyperparameter Tuning:

I used the following search space for hyperparameter tuning:

<img width="768" height="357" alt="image" src="https://github.com/user-attachments/assets/3f1c440e-b4cc-4c31-aba4-34bc2254c8be" />

And I used the BayesSearchCV() for finding the optimal model, with cv = 5. 100 iterations, and 1000 n-estimators shown above. As per the assignment, the scoring was done using RMSE:

<img width="764" height="87" alt="image" src="https://github.com/user-attachments/assets/bd42fca2-06a9-4b07-b968-c4533f4f01b3" />

Finally, fitting the optimal model and saving the prediction for submission:

<img width="559" height="267" alt="image" src="https://github.com/user-attachments/assets/ec08a817-c94b-410d-8cc4-008a8b5d2c82" />

## Kaggle Score:

The lowest score I could get on Kaggle was 128.3, which is a big improvement from the first submission of 800

<img width="940" height="68" alt="image" src="https://github.com/user-attachments/assets/0a29958c-6992-4136-97ac-6c1a3f7079c8" />

## Further Improvements:
I wanted to implement lagging/rolling features, but the dataset is missing many hours. This could be corrected by either adding the missing hours and filling in the missing data with mean values/forward filling for categorical data or converting hourly data to daily data. In this assignment, I had to submit all 1752 columns given in the test set, so this was not possible to implement. 
I would also like try other regression models like Lasso, Ridge, LightGBM, etc. 

Thank you for checking out my program!
