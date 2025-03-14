# %% [markdown]
# # **Flight Fare Prediction**

# %%


# %% [markdown]
# # Importing Libraries

# %%
# Basic Libraries
import pandas as pd
import numpy as np
# Visualization Libraries
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

import seaborn as sns
# Suppress specific FutureWarnings
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR  # Importing Support Vector Regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle

# %% [markdown]
# # Loading The Flight Fare Data

# %% [markdown]
# Importing Dataset

# %%
Train_data=pd.read_excel("C:/Users/lenovo/OneDrive/Desktop/Project_IV(sem)/Data_Train.xlsx")

# %% [markdown]
# 

# %% [markdown]
# # Data Inspection

# %%
# first five rows of the dataset?
Train_data.head(5)

# %% [markdown]
# #### Shape of the dataset

# %%
Train_data.shape

# %% [markdown]
# #### Summary of the dataset

# %%
Train_data.describe()

# %% [markdown]
# Since this dataset has only 1 numercial column, describe() will display summary statistics only for the "Price" column.

# %%
#information about the data, checking datatypes

Train_data.info()

# %% [markdown]
# ## Train Data has only 1 null value in route and total_stops

# %% [markdown]
# ## Dataset Overview
# - Total Entries: 10,683 (rows) and 11 features in train data
# - Toatl entries in test data 2671
# - Total Columns: 11
# - Key Columns:
# - Airline: Categorical
# - Date_of_Journey: Object (to be converted to datetime)
# - Source: Categorical
# - Destination: Categorical
# - Route: Object (1 missing value)
# - Dep_Time: Object (to be converted to datetime)
# - Arrival_Time: Object (to be converted to datetime)
# - Duration: Object (to be cleaned and possibly converted)
# - Total_Stops: Object (1 missing value)
# - Additional_Info: Object
# - Price: Numeric (int64)
# - Missing Values:
# - Route: 1 missing value
# - Total_Stops: 1 missing value
# - Data Type Conversions Needed:
#  - Convert Date_of_Journey, Dep_Time, and Arrival_Time to appropriate datetime formats.
#  - Consider converting Total_Stops to numeric or categorical as needed.

# %% [markdown]
# ### Exploratory Data Analysis (EDA)

# %% [markdown]
# #### Bar chart showing top 10 most preferred Airlines

# %%
plt.figure(figsize=(12,5))
sns.countplot(x="Airline", data=Train_data,order = Train_data['Airline'].value_counts().index,ec = "black")
font_style={'family':'times new roman','size':20,'color':'black'}
plt.title("Most preferred Airlines",fontdict=font_style)
plt.ylabel("Count",fontdict=font_style)
plt.xlabel("Airlines",fontdict=font_style)
plt.xticks(rotation= 90)
plt.xlim(-1,10.5)
plt.show()

# %% [markdown]
# ##### Insights
# - Most preferred Airline is "Jet Airways"
# - From all the total flight tickets sold, Jet Airways has the highest share followed by Indigo.

# %% [markdown]
# #### Airlines Vs Flight ticket Price
# 

# %%
airlines = Train_data.groupby('Airline').Price.max()
airlines_df= airlines.to_frame().sort_values('Price',ascending=False)[0:10]
airlines_df

# %%
plt.subplots(figsize=(12,5))
sns.barplot(x=airlines_df.index, y=airlines_df["Price"],ec = "black")
font_style={'family':'times new roman','size':20,'color':'black'}
plt.title("Airlines Company vs Flight Ticket Price",fontdict=font_style )
plt.ylabel("Flight Ticket Price", fontdict=font_style)
plt.xlabel("Airlines", fontdict=font_style)
plt.xticks(rotation=90)
plt.show()

# %% [markdown]
# ##### Insights
# - "Jet Airways Business" tickets are the most expensive ones.

# %% [markdown]
# #### Price on Weekdays vs Weekends

# %%
days_df = Train_data[['Airline','Date_of_Journey', 'Price']].copy()
days_df.head()

# %%
days_df['Date_of_Journey'] = pd.to_datetime(days_df['Date_of_Journey'],format='%d/%m/%Y')
days_df['Weekday'] = days_df['Date_of_Journey'].dt.day_name()
days_df['Weekend'] = days_df['Weekday'].apply(lambda day: 1 if day == 'Sunday' else 0)
days_df.head()

# %%
# Check and convert Weekend to categorical
if days_df['Weekend'].dtype != 'category':
    days_df['Weekend'] = days_df['Weekend'].map({0: 'Weekday', 1: 'Weekend'}).astype('category')

# Plotting
plt.subplots(figsize=(12, 5))
sns.barplot(data=days_df, x='Airline', y='Price', hue='Weekend')
plt.xlabel("Airline", size=15)
plt.xticks(rotation=90)
plt.title('Average Price by Airline and Weekend/Weekday')
plt.legend(title='Day Type')  # Optional
plt.show()


# %% [markdown]
# ##### Insights
# - The Price of tickets is higher on Weekends.

# %%
# Boxplot for Price by Total Stops
plt.figure(figsize=(10, 6))
sns.boxplot(data=Train_data, x='Total_Stops', y='Price')
plt.title('Flight Prices by Number of Stops')
plt.show()

# Creating a summary DataFrame
price_summary = Train_data.groupby('Total_Stops')['Price'].agg(['mean',  'min', 'max']).reset_index()
price_summary.columns = ['Total_Stops', 'Mean_Price', 'Min_Price', 'Max_Price']

# Display the summary DataFrame
price_summary


# %% [markdown]
# ### Insights

# %% [markdown]
# - More stops, higher prices: The average price tends to increase as the number of stops increases,
# - but non-stop flights remain the most economical option overall.

# %% [markdown]
# ## Data Preprocessing

# %% [markdown]
# #### Data cleaning --> Outlier Detection

# %%
Q1 = Train_data['Price'].quantile(0.25)
Q3 = Train_data['Price'].quantile(0.75)
IQR = Q3 - Q1

# Determine outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = Train_data[(Train_data['Price'] < lower_bound) | (Train_data['Price'] > upper_bound)]
print("Outliers in Flight Prices:\n")
outliers

# %% [markdown]
# ### Removing Outlier

# %%
# Remove outliers from the dataset
Train_data_cleaned = Train_data[~Train_data['Price'].isin(outliers['Price'])]


# Verify the shape of the new dataset
print("Original Data Shape:", Train_data.shape)
print("Cleaned Data Shape:", Train_data_cleaned.shape)


# %% [markdown]
# ## Check for duplicate rows

# %%
duplicate_rows = Train_data_cleaned[Train_data_cleaned.duplicated()]

# Display duplicate rows
print("Duplicate Rows:\n")
duplicate_rows


# %%
#number of duplicate rows
print("Number of duplicate rows:", duplicate_rows.shape[0])

# %%
# Keep only the first occurrence of duplicates
Train_data_cleaned = Train_data_cleaned.drop_duplicates(keep='first')

# %%
#after removing duplicate rows and outlier total number of rows
Train_data_cleaned.shape

# %% [markdown]
# ### Feature Engineering on Train data

# %%
# Extact day, month, year from Date_of_Journey feature and store them in new columns.
Train_data_cleaned["Journey_date"]=Train_data_cleaned["Date_of_Journey"].str.split("/").str[0].astype(int)
Train_data_cleaned["Journey_month"]=Train_data_cleaned["Date_of_Journey"].str.split("/").str[1].astype(int)
Train_data_cleaned["Journey_year"]=Train_data_cleaned["Date_of_Journey"].str.split("/").str[2].astype(int)

# %%
Train_data_cleaned.head()

# %%
# Now Date_of_Journey column is no longer required, so we can drop it.
Train_data_cleaned=Train_data_cleaned.drop(["Date_of_Journey"],axis=1)

# %%
Train_data_cleaned["Journey_year"].value_counts()

# %%
# Since Journey_year is the same ("2019") for all rows, we can drop it.
Train_data_cleaned=Train_data_cleaned.drop(["Journey_year"],axis=1)
Train_data_cleaned.head()

# %%
# Total_Stops
Train_data_cleaned["Total_Stops"]=Train_data_cleaned["Total_Stops"].str.split(" ").str[0]
Train_data_cleaned["Total_Stops"]=Train_data_cleaned["Total_Stops"].replace("non-stop","0")
Train_data_cleaned.head()

# %%
# Total stops is object datatype till

# %%
# Extracting hours and min from Arrival time, Departure time
# Arrival_Time
Train_data_cleaned["Arrival_Time"]=Train_data_cleaned["Arrival_Time"].str.split(" ").str[0]
Train_data_cleaned['Arrival_hour']=Train_data_cleaned["Arrival_Time"].str.split(':').str[0].astype(int)
Train_data_cleaned['Arrival_min']=Train_data_cleaned["Arrival_Time"].str.split(':').str[1].astype(int)
Train_data_cleaned=Train_data_cleaned.drop(["Arrival_Time"],axis=1)

# %%
# Dep_Time
Train_data_cleaned['Dep_hour']=Train_data_cleaned["Dep_Time"].str.split(':').str[0].astype(int)
Train_data_cleaned['Dep_min']=Train_data_cleaned["Dep_Time"].str.split(':').str[1].astype(int)
Train_data_cleaned=Train_data_cleaned.drop(["Dep_Time"],axis=1)

Train_data_cleaned.head()

# %% [markdown]
# #### Route column
# 
# - The route column tells about the journey's path.
# - Route column can be removed because 'Total_Stops' field has already captured this value and both are related.
# 
# 'Additional_Info' column can be dropped since more than 70% of them have no information.

# %%
Train_data_cleaned.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

# %% [markdown]
# #### Duration column
# - Extract hours and min from Duration feature.
# - Convert hours to min and find total duration in minutes to make it a single column.

# %%
#making column duration_hr
Train_data_cleaned["Duration_hr"]=Train_data_cleaned["Duration"].str.split(' ').str[0].str.split('h').str[0]

# %%
#construct column duration_min
Train_data_cleaned["Duration_min"]=Train_data_cleaned["Duration"].str.split(' ').str[1].str.split('m').str[0]

# %%
Train_data_cleaned.head()

# %%
print("no of null value in  duration min ",Train_data_cleaned['Duration_min'].isnull().sum())
#FILL IT WITH 0
Train_data_cleaned['Duration_min'].fillna("0",inplace=True)

# %%
Train_data_cleaned[Train_data_cleaned['Duration_hr'].str.contains('m')]

# %%
#now we shift the minute value from  duration hour to  duration min
#replace the value of duration hour with 0

# %%
Train_data_cleaned.loc[[6474], 'Duration_min'] = Train_data_cleaned.loc[[6474], 'Duration_hr']
Train_data_cleaned.loc[Train_data_cleaned['Duration_min'] == '5m', 'Duration_min'] = 5
Train_data_cleaned["Duration_hr"]=Train_data_cleaned["Duration_hr"].replace("5m","0")

# %%
#now convert the duration hour and minute  datatype to integer
Train_data_cleaned["Duration_min"] = Train_data_cleaned["Duration_min"].astype(int)
Train_data_cleaned["Duration_hr"] = Train_data_cleaned["Duration_hr"].astype(int)

# %%
#create a new colum with the name of duaration
Train_data_cleaned["Duration"] = (Train_data_cleaned["Duration_hr"]*60) + Train_data_cleaned["Duration_min"]
Train_data_cleaned=Train_data_cleaned.drop(['Duration_hr','Duration_min'],axis=1)
Train_data_cleaned.head()

# %% [markdown]
# ### Checking for null values

# %%
Train_data_cleaned.isnull().sum()

# %%
# filling Total_Stops
Train_data_cleaned["Total_Stops"].value_counts()

# %%
#all column convert to the correct form
Train_data_cleaned.dtypes

# %%
# '1' is most frequently occuring value. So fill Total_Stops column null values by '1'
Train_data_cleaned["Total_Stops"]=Train_data_cleaned["Total_Stops"].fillna('1')

# %%
Train_data_cleaned["Total_Stops"]=Train_data_cleaned["Total_Stops"].astype(int)

# %%
Train_data_cleaned.isnull().sum()

# %%
Train_data_cleaned.shape

# %% [markdown]
# ### there is no null values now in the data set

# %% [markdown]
# ### Using Encoding to Handle categorical data
# ##### Features with Categorical data
# - Airline
# - Source
# - Destination
# 
# Apply Label Encoder to these features.

# %%
from sklearn.preprocessing import LabelEncoder
la=LabelEncoder()
for i in ["Airline","Source","Destination"]:
    Train_data_cleaned[i]=la.fit_transform(Train_data_cleaned[i])
Train_data_cleaned.head()

# %% [markdown]
# ### Feature Selection

# %% [markdown]
# - Finding out the best feature which will contribute and have good relation with the target variable(Price).
# - Let's use heat map to find correlation between features.

# %%
plt.figure(figsize=(15,10))

sns.heatmap(Train_data_cleaned.corr(),annot=True,cmap='RdYlGn')
plt.title("Heat map showing Correlation between features")

plt.show()

# %% [markdown]
# #### There is a high correlation between:
# - Price & Total Stops
# - Price & flight duration
# - Duration & Total stops

# %% [markdown]
# #### now we prepare our test data which is given

# %% [markdown]
# ### Loading the  Test Flight Fare Data
# - Importing dataset
# - Since data is in form of excel file we have to use pandas read_excel to load the data

# %%
Test_data=pd.read_excel("C:/Users/lenovo/OneDrive/Desktop/Project_IV(sem)/Test_set.xlsx")
Test_data.shape

# %%
#as it i a test data there is no price column

# %% [markdown]
# ### Summary of the dataset

# %%
Test_data.describe()

# %%
#information about the data, checking datatypes

Test_data.info()

# %% [markdown]
# ## Dataset Overview
# - Total Entries: 2671 (rows) and 10 features in test data
# - Toatl entries in test data 2671
# - Total Columns: 10
# - Key Columns:
# - Airline: Categorical
# - Date_of_Journey: Object (to be converted to datetime)
# - Source: Categorical
# - Destination: Categorical
# - Route: Object (1 missing value)
# - Dep_Time: Object (to be converted to datetime)
# - Arrival_Time: Object (to be converted to datetime)
# - Duration: Object (to be cleaned and possibly converted)
# - Total_Stops: Object (1 missing value)
# - Additional_Info: Object
# - Price has to predict
# - Data Type Conversions Needed:
# - Convert Date_of_Journey, Dep_Time, and Arrival_Time to appropriate datetime formats.
# - Consider converting Total_Stops to numeric or categorical as needed.

# %% [markdown]
# ## Feature Engineering on Test data

# %%
# Extact day, month, year from Date_of_Journey feature and store them in new columns.
Test_data["Journey_date"]=Test_data["Date_of_Journey"].str.split("/").str[0].astype(int)
Test_data["Journey_month"]=Test_data["Date_of_Journey"].str.split("/").str[1].astype(int)
Test_data["Journey_year"]=Test_data["Date_of_Journey"].str.split("/").str[2].astype(int)

# %%
Test_data.head()

# %%
# Now Date_of_Journey column is no longer required, so we can drop it.
Test_data=Test_data.drop(["Date_of_Journey"],axis=1)

# %%
Test_data["Journey_year"].value_counts()

# %%
#Since Journey_year is the same ("2019") for all rows, we can drop it.
Test_data=Test_data.drop(["Journey_year"],axis=1)
Test_data.head()

# %%
# Total_Stops
Test_data["Total_Stops"]=Test_data["Total_Stops"].str.split(" ").str[0]
Test_data["Total_Stops"]=Test_data["Total_Stops"].replace("non-stop","0")
Test_data.head()

# %%
# Total stops is object datatype till

# %%
# Extracting hours and min from Arrival time, Departure time
# Arrival_Time
Test_data["Arrival_Time"]=Test_data["Arrival_Time"].str.split(" ").str[0]
Test_data['Arrival_hour']=Test_data["Arrival_Time"].str.split(':').str[0].astype(int)
Test_data['Arrival_min']=Test_data["Arrival_Time"].str.split(':').str[1].astype(int)
Test_data=Test_data.drop(["Arrival_Time"],axis=1)

# %%
# Dep_Time
Test_data['Dep_hour']=Test_data["Dep_Time"].str.split(':').str[0].astype(int)
Test_data['Dep_min']=Test_data["Dep_Time"].str.split(':').str[1].astype(int)
Test_data=Test_data.drop(["Dep_Time"],axis=1)

Test_data.head()

# %% [markdown]
# #### Route column
# 
# - The route column tells about the journey's path.
# - Route column can be removed because 'Total_Stops' field has already captured this value and both are related.
# 
# 'Additional_Info' column can be dropped since more than 70% of them have no information.

# %%
Test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

# %% [markdown]
# #### Duration column
# - Extract hours and min from Duration feature.
# - Convert hours to min and find total duration in minutes to make it a single column.

# %%
#making column duration_hr
Test_data["Duration_hr"]=Test_data["Duration"].str.split(' ').str[0].str.split('h').str[0]

# %%
#construct column duration_min
Test_data["Duration_min"]=Test_data["Duration"].str.split(' ').str[1].str.split('m').str[0]

# %%
Test_data.head()

# %%
print("no of null value in  duration min ",Test_data['Duration_min'].isnull().sum())
#FILL IT WITH 0
Test_data['Duration_min'].fillna("0",inplace=True)

# %%
Test_data[Test_data['Duration_hr'].str.contains('m')]

# %% [markdown]
# - now we shift the minute value from  duration hour to  duration min
# - replace the value of duration hour with 0

# %%
Test_data.loc[[2660], 'Duration_min'] = Test_data.loc[[2660], 'Duration_hr']
Test_data.loc[Test_data['Duration_min'] == '5m', 'Duration_min'] = 5
Test_data["Duration_hr"]=Test_data["Duration_hr"].replace("5m","0")

# %%
#now convert the duration hour and minute  datatype to integer
Test_data["Duration_min"] = Test_data["Duration_min"].astype(int)
Test_data["Duration_hr"] = Test_data["Duration_hr"].astype(int)

# %%
#create a new colum with the name of duaration
Test_data["Duration"] = (Test_data["Duration_hr"]*60) + Test_data["Duration_min"]
Test_data=Test_data.drop(['Duration_hr','Duration_min'],axis=1)
Test_data.head()

# %% [markdown]
# ### Checking for null values test data
# 

# %%
Test_data.isnull().sum()

# %%
#all column convert to the correct form
Test_data.dtypes

# %%
Test_data["Total_Stops"]=Test_data["Total_Stops"].astype(int)

# %%
Test_data.shape

# %%
Test_data.info()

# %% [markdown]
# ### Using Encoding to Handle categorical data
# ##### Features with Categorical data
# - Airline
# - Source
# - Destination
# 
# Apply Label Encoder to these features.

# %%
from sklearn.preprocessing import LabelEncoder
la=LabelEncoder()
for i in ["Airline","Source","Destination"]:
    Test_data[i]=la.fit_transform(Test_data[i])
Test_data.head()

# %% [markdown]
# ### Building Machine Learning models

# %% [markdown]
# For predicting the Price, we build 3 models using the following algorithms:
# - Linear Regression
# - Decision Tree regressor
# - Random Forest Regressor
# 
# Compare the accuracies got from these 3 models and select the best model.
# Apply hyperparameter tuning to increase its efficiency.
#         

# %%
#splitting data into train and test dataframe
train_df=Train_data_cleaned

# %%
print(train_df.shape)
print(Test_data.shape)

# %%
#splitting data into x and y
x=train_df.drop(["Price"],axis=1)
y=train_df.loc[:,["Price"]].values

# %%
# spiliting the dataset into train data and test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=25)

# %% [markdown]
# #### Import models

# %%

# Create a DataFrame to store results
results = pd.DataFrame(columns=['Model', 'R² Score', 'MAE', 'MSE'])

# %%
# Function to train the models and evaluate
def predict(algorithm):
    print("Algorithm:", algorithm.__class__.__name__)  # Print the name of the algorithm
    model = algorithm.fit(x_train, y_train)  # Train the model
    y_pred = model.predict(x_test)  # Make predictions

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)  # Calculate R² score
    mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
    mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
    results.loc[len(results)] = [algorithm.__class__.__name__, r2, mae, mse]

# %%
# Example usage of the function with different algorithms
algorithms = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    SVR(kernel='linear')  # Using linear kernel for SVR
]

# %% [markdown]
# ### Scale the features ( standardization) fro svr

# %%
# Scale features for SVR
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# %%
# Loop through the algorithms and make predictions
for algo in algorithms:
    if algo.__class__.__name__ == 'SVR':
        # Use the scaled features for SVR
        predict(algo.fit(x_train_scaled, y_train))
    else:
        predict(algo)

# Display the results DataFrame
print("\nModel Evaluation Results:")
results

# %% [markdown]
# The **Random Forest Regressor model** turned out to be the most accurate one out of the 3 models.
# 
# Let's try to improve the accuracy by doing Hyperparameter tuning.

# %% [markdown]
# ### Hyperparameter tuning

# %% [markdown]
# #### Applying  Random Search to find the best parameters

# %%
from sklearn.model_selection import RandomizedSearchCV
random_search = {'n_estimators' : [100, 120, 150, 180, 200,220,250],
                 'max_features':['auto','sqrt'],
                 'max_depth':[5,10,15,20],
                 'min_samples_split' : [2, 5, 10, 15, 100],
                 'min_samples_leaf' : [1, 2, 5, 10]}
rf_regressor=RandomForestRegressor()
rf_model=RandomizedSearchCV(estimator=rf_regressor,param_distributions=random_search,
                            cv=3,n_jobs=-1,verbose=2,random_state=0)
rf_model.fit(x_train,y_train)

# %%
# best parameter
rf_model.best_params_

# %%
#predicting the values
pred=rf_model.predict(x_test)
r2_score(y_test,pred)

# %% [markdown]
# #### After hypertuning, the accuracy increases.

# %%
print('r2_score:',r2_score(y_test,pred))
print('MAE:', mean_absolute_error(y_test, pred))
print('MSE:', mean_squared_error(y_test, pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, pred)))

# %% [markdown]
# # For Random Forest Regressor,
# - Before hyper tuning, R2 score = 81 %
# - After hyper tuning, R2 score = 83 %
# 

# %% [markdown]
# ##  prediction on Test data

# %%
test_predictions = rf_model.predict(Test_data)
# As we have done same feature engineering

# %%
submission = pd.DataFrame({ 'Price': test_predictions})
submission.to_csv('submission.csv', index=False)

# %% [markdown]
# ### save the model

# %%
from pickle import dump
dump(rf_model,open('flightfare.pkl','wb'))

# %% [markdown]
# ##### Conclusion
# 
# We have used random forest regressor for training the model and improved its accuracy by doing hyperparameter tuning. As a result, we have trained our **Random Forest Regression model**, to forecast fares of flight tickets, with an R2 score of 83 %.


