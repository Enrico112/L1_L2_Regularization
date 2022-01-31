import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import warnings

# suppress warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('melbourne_housing.csv')
df.shape
df.nunique()

# retain cols to use
cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount',
               'Distance', 'CouncilArea', 'Bedroom2','Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']
df = df[cols_to_use]

# count nas in every col
df.isna().sum()

# cols to fill with zero
cols_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
df[cols_fill_zero] = df[cols_fill_zero].fillna(0)

# cols to fill with mean
df['Landsize'] = df['Landsize'].fillna(df.Landsize.mean())
df['BuildingArea'] = df['BuildingArea'].fillna(df.Landsize.mean())

# drop remaining nas in price
df.dropna(inplace=True)

# create dummies for categorical vars, drop one redundant column
df = pd.get_dummies(df, drop_first=True)

# create X an y datasets
X = df.drop('Price', axis=1)
y = df['Price']

# split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# fit linear reg
linreg = LinearRegression().fit(X_train, y_train)
# test model
linreg.score(X_test,y_test)
# score is low

# regularize with Lasso (L1), tolerance=.1
lasso = Lasso(alpha=80, max_iter=1000, tol=.5)
lasso.fit(X_train, y_train)
lasso.score(X_test,y_test)

# regularize with ridge (L2)
ridge = Ridge(alpha=80, max_iter=100, tol=.5)
ridge.fit(X_train, y_train)
ridge.score(X_test,y_test)