import pandas as pd
import numpy as np

# Load Excel files
city = pd.read_excel(r"C:\Users\Karth\OneDrive\Desktop\PROJECT 4\Tourism Dataset\City.xlsx")
continent = pd.read_excel(r"C:\Users\Karth\OneDrive\Desktop\PROJECT 4\Tourism Dataset\Continent.xlsx")
country = pd.read_excel(r"C:\Users\Karth\OneDrive\Desktop\PROJECT 4\Tourism Dataset\Country.xlsx")
item = pd.read_excel(r"C:\Users\Karth\OneDrive\Desktop\PROJECT 4\Tourism Dataset\Item.xlsx")
mode = pd.read_excel(r"C:\Users\Karth\OneDrive\Desktop\PROJECT 4\Tourism Dataset\Mode.xlsx")
region = pd.read_excel(r"C:\Users\Karth\OneDrive\Desktop\PROJECT 4\Tourism Dataset\Region.xlsx")
transaction = pd.read_excel(r"C:\Users\Karth\OneDrive\Desktop\PROJECT 4\Tourism Dataset\Transaction.xlsx")
user = pd.read_excel(r"C:\Users\Karth\OneDrive\Desktop\PROJECT 4\Tourism Dataset\User.xlsx")
type_df = pd.read_excel(r"C:\Users\Karth\OneDrive\Desktop\PROJECT 4\Tourism Dataset\Type.xlsx")
attraction=pd.read_excel(r"C:\Users\Karth\OneDrive\Desktop\PROJECT 4\Tourism Dataset\Updated_Item.xlsx")

#Data cleaning
mode.drop(index=0,inplace=True)
city.fillna('unknown',inplace=True)
continent.drop(index=0, inplace=True)
country.drop(index=0, inplace=True)
region.drop(index=0, inplace=True)
user.fillna("0", inplace=True)
# Remove duplicates
city.drop_duplicates(inplace=True)
continent.drop_duplicates(inplace=True)
country.drop_duplicates(inplace=True)
region.drop_duplicates(inplace=True)
transaction.drop_duplicates(inplace=True)
user.drop_duplicates(inplace=True)
item.drop_duplicates(inplace=True)
mode.drop_duplicates(inplace=True)
type_df.drop_duplicates(inplace=True)

# Check for valid foreign keys
print(transaction['UserId'].isin(user['UserId']).all())
print(transaction['AttractionId'].isin(item['AttractionId']).all())


#Merging the dataset:
df = transaction.merge(user, on='UserId', how='left')

df['CityId'] = df['CityId'].astype(int)
city['CityId'] = city['CityId'].astype(int)
df = df.merge(city, on='CityId', how='left')

df = df.merge(country, left_on='CountryId_x', right_on='CountryId', how='left')

df.drop(columns=['CountryId_y', 'CountryId'], inplace=True)
df.rename(columns={'CountryId_x': 'CountryId'}, inplace=True)

df = df.merge(region, left_on='RegionId_x', right_on='RegionId', how='left')
df.drop(columns=['RegionId_y','RegionId'], inplace=True)
df.rename(columns={'RegionId_x': 'RegionId'}, inplace=True)

df = df.merge(continent, left_on='ContinentId_x', right_on='ContinentId', how='left')
df.drop(columns=['ContinentId_y', 'ContinentId'], inplace=True)
df.rename(columns={'ContinentId_x': 'ContinentId'}, inplace=True)

df = df.merge(item, on='AttractionId', how='left')
df['AttractionTypeId'] = df['AttractionTypeId'].astype(int)
df = df.merge(type_df, on='AttractionTypeId', how='left')

df['VisitMode'] = df['VisitMode'].astype(str)
mode['VisitModeId'] = mode['VisitModeId'].astype(str)

df = df.merge(mode, left_on='VisitMode', right_on='VisitModeId', how='left')
df.drop(columns=['VisitMode_x'], inplace=True)
df.rename(columns={'VisitMode_y': 'VisitMode'}, inplace=True)

df #printing the dataset:
df.info()

#EDA ->
# User Distribution Across Continents, Countries, and Regions
import matplotlib.pyplot as plt
import seaborn as sns
# Users by Continent
continent_dist = df['ContinentId'].value_counts()
sns.barplot(x=continent_dist.index, y=continent_dist.values)
plt.title('User Distribution by Continent')
plt.xlabel('ContinentId')
plt.ylabel('Number of Users')
plt.show()
# Users by Country
country_dist = df['CountryId'].value_counts().head(10)
sns.barplot(x=country_dist.index, y=country_dist.values)
plt.title('Top 10 Countries by User Count')
plt.xlabel('CountryId')
plt.ylabel('Number of Users')
plt.xticks(rotation=45)
plt.show()

# Attraction Types & Popularity Based on Ratings
# Average rating per attraction type
attraction_popularity = df.groupby('AttractionTypeId')['Rating'].agg(['count', 'mean']).reset_index()
attraction_popularity = attraction_popularity.sort_values('count', ascending=False)
# Bar plot for count of ratings by type
sns.barplot(x='AttractionTypeId', y='count', data=attraction_popularity)
plt.title('Attraction Popularity by Number of Ratings')
plt.xlabel('AttractionTypeId')
plt.ylabel('Number of Ratings')
plt.show()
# Bar plot for average rating by type
sns.barplot(x='AttractionTypeId', y='mean', data=attraction_popularity)
plt.title('Average Rating by Attraction Type')
plt.xlabel('AttractionTypeId')
plt.ylabel('Average Rating')
plt.show()

# Correlation Between VisitMode and Demographics (e.g., ContinentId, CountryId) 
# Visit mode vs continent
visit_continent = pd.crosstab(df['VisitMode'], df['ContinentId'], normalize='index')
visit_continent.plot(kind='bar', stacked=True, cmap='tab20')
plt.title('Visit Mode vs Continent')
plt.xlabel('Visit Mode')
plt.ylabel('Proportion of Users')
plt.legend(title='ContinentId', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# Visit mode vs country (Top 5 countries)
top_countries = df['CountryId'].value_counts().head(5).index
subset = df[df['CountryId'].isin(top_countries)]
visit_country = pd.crosstab(subset['VisitMode'], subset['CountryId'], normalize='index')
visit_country.plot(kind='bar', stacked=True, cmap='Set3')
plt.title('Visit Mode vs Top 5 Countries')
plt.xlabel('Visit Mode')
plt.ylabel('Proportion of Users')
plt.legend(title='CountryId')
plt.show()

# Distribution of Ratings Across Attractions and Regions
# Histogram of ratings
sns.histplot(df['Rating'], bins=20, kde=True)
plt.title('Distribution of User Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()
# Boxplot: Ratings per Attraction
top_attractions = df['AttractionId'].value_counts().head(10).index
sns.boxplot(x='AttractionId', y='Rating', data=df[df['AttractionId'].isin(top_attractions)])
plt.title('Rating Distribution for Top 10 Attractions')
plt.xlabel('AttractionId')
plt.ylabel('Rating')
plt.xticks(rotation=45)
plt.show()
# Region-wise rating distribution if RegionId exists
if 'RegionId' in df.columns:
    region_ratings = df.groupby('RegionId')['Rating'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=region_ratings.index, y=region_ratings.values)
    plt.title('Average Rating by Region')
    plt.xlabel('RegionId')
    plt.ylabel('Average Rating')
    plt.show()


# Model Training:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error,root_mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 1. Basic Feature Engineering
df['UserAvgRating'] = df.groupby('UserId')['Rating'].transform('mean')
df['UserReviewCount'] = df.groupby('UserId')['Rating'].transform('count')
df['AttractionAvgRating'] = df.groupby('AttractionId')['Rating'].transform('mean')
df['AttractionReviewCount'] = df.groupby('AttractionId')['Rating'].transform('count')

# 2. Prepare Features and Target
features = [
    'VisitYear', 'VisitMonth', 'ContinentId', 'CountryId',
    'AttractionTypeId','UserReviewCount','AttractionAvgRating','AttractionReviewCount',
      'VisitMode','UserAvgRating'  
]
target = 'Rating'
X = df[features]
y = df[target]

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
numeric_features = ['VisitYear', 'VisitMonth', 'UserAvgRating', 'UserReviewCount','AttractionAvgRating','AttractionReviewCount']
categorical_features = ['ContinentId', 'CountryId', 'AttractionTypeId', 'VisitMode']
preprocessor = ColumnTransformer(
    transformers=[
        ('poly', Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=4, include_bias=False))
        ]), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 5. Model Pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 6. Train Model
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")

y_pred = model.predict(X_train)
print(f"R2 Score: {r2_score(y_train, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_train, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_train, y_pred):.4f}")
print(f"RMSE: {root_mean_squared_error(y_train, y_pred):.4f}")

#CLASSIFICATION:
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
clf_ct = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=15,
    min_samples_split=10,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced' ))
])
clf_ct.fit(X_train, y_train)
y_pred_ct = clf_ct.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_ct))
print("Random Forest Accuracy score:")
print(accuracy_score(y_test,y_pred_ct))
print("Random Forest Confussion matrix:")
print(confusion_matrix (y_test,y_pred_ct))

y_pred_ct = clf_ct.predict(X_train)
print("Random Forest Classification Report:")
print(classification_report(y_train, y_pred_ct))
print("Random Forest Accuracy score:")
print(accuracy_score(y_train,y_pred_ct))
print("Random Forest Confussion matrix:")
print(confusion_matrix (y_train,y_pred_ct))

from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 15, 20],
    'classifier__min_samples_split': [5, 10],
    'classifier__min_samples_leaf': [2, 4]
}

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=clf_ct,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy', 
    verbose=2,
    n_jobs=-1
)

# Fit GridSearch
grid_search.fit(X_train, y_train)
# Print best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)
# Predict using the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)


clf_ct.fit(X_train, y_train)
y_pred_ct = clf_ct.predict(X_test)
print("Classification Report (Test):")
print(classification_report(y_test, y_pred_best))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))
print("Test Accuracy:", accuracy_score(y_test, y_pred_best))

import mysql.connector

# Connect to MySQL server (without specifying the database)
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Sk@112308",
)

cursor = conn.cursor()

# Create the database if it doesn't exist
create_db_query = "CREATE DATABASE IF NOT EXISTS tourism_experience"
cursor.execute(create_db_query)

# Close the initial connection
cursor.close()
conn.close()

# Now connect to the specific database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Sk@112308",
    database="tourism_experience",  # Now the database exists
)

cursor = conn.cursor()
# Create table (if not exists)
create_table_query = """
CREATE TABLE IF NOT EXISTS TourismRatings (
    TransactionId INT,
    UserId INT,
    VisitYear INT,
    VisitMonth INT,
    AttractionId INT,
    Rating INT,
    ContinentId INT,
    RegionId INT,
    CountryId INT,
    CityId INT,
    Country VARCHAR(100),
    Region VARCHAR(100),
    Continent VARCHAR(100),
    AttractionCityId INT,
    AttractionTypeId INT,
    Attraction VARCHAR(255),
    AttractionAddress VARCHAR(255),
    AttractionType VARCHAR(100),
    VisitModeId INT,
    VisitMode VARCHAR(50)
);
"""
cursor.execute(create_table_query)

# Insert data from DataFrame into SQL
insert_query = """
INSERT INTO TourismRatings (
    TransactionId, UserId, VisitYear, VisitMonth, AttractionId, Rating,
    ContinentId, RegionId, CountryId, CityId,
    Country, Region, Continent,
    AttractionCityId, AttractionTypeId, Attraction, AttractionAddress, AttractionType,
    VisitModeId, VisitMode
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

# Convert DataFrame rows to list of tuples and insert
data_to_insert = df[[  # Adjust the DataFrame variable name if needed
    'TransactionId', 'UserId', 'VisitYear', 'VisitMonth', 'AttractionId', 'Rating',
    'ContinentId', 'RegionId', 'CountryId', 'CityId',
    'Country', 'Region', 'Continent',
    'AttractionCityId', 'AttractionTypeId', 'Attraction', 'AttractionAddress', 'AttractionType',
    'VisitModeId', 'VisitMode'
]].where(pd.notnull(df), None).values.tolist()

cursor.executemany(insert_query, data_to_insert)
conn.commit()

print(f"{cursor.rowcount} rows were inserted.")

# Close connection
cursor.close()
conn.close()