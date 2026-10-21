import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
matplotlib.use('TkAgg', force = True)

pd.set_option('display.max_columns', None)
df = pd.read_csv("C:/Bank Customer Churn Prediction.csv")
df.head()

print(df.info())

print(df.describe())

print(df.isnull().sum())

df = df.drop("customer_id", axis=1)

numeric_columns = [
   "credit_score",
   "age",
   "tenure",
   "balance",
   "products_number",
   "credit_card",
   "active_member",
   "estimated_salary",
   "churn"
]


numeric_data = df[numeric_columns]
summary = numeric_data.describe()
summary.loc["median"] = numeric_data.median()
print(summary)


for col in numeric_columns:
   plt.figure(figsize=(6, 4))
   sns.histplot(df[col], kde=True, bins=30)
   plt.title(f"Распределение признака {col}")
   plt.xlabel(col)
   plt.ylabel("Частота")
   plt.grid()
   plt.show()

print(df["churn"].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="churn")
plt.title("Распределение клиентов по признаку churn")
plt.xlabel("0 — клиент остался, 1 — клиент ушел")
plt.ylabel("Количество клиентов")
plt.grid(axis="y")
plt.show()

def calculate_metrics(y_test, y_pred):
   mae = mean_absolute_error(y_test, y_pred)
   mse = mean_squared_error(y_test, y_pred)
   rmse = np.sqrt(mse)
   r2 = r2_score(y_test, y_pred)
   return mae, mse, rmse, r2

X = df[["age"]]
y = df["balance"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


simple_model = LinearRegression()
simple_model.fit(X_train, y_train)


y_pred_simple = simple_model.predict(X_test)


print("Парная линейная регрессия")
print("Коэффициент:", simple_model.coef_)
print("Свободный член:", simple_model.intercept_)


mae_simple, mse_simple, rmse_simple, r2_simple = calculate_metrics(y_test, y_pred_simple)


print("MAE:", mae_simple)
print("MSE:", mse_simple)
print("RMSE:", rmse_simple)
print("R2:", r2_simple)


plt.figure(figsize=(7, 5))
plt.scatter(X_test, y_test, alpha=0.6, label="Фактические значения")
plt.plot(X_test, y_pred_simple, color="red", label="Линия регрессии")
plt.xlabel("Возраст клиента")
plt.ylabel("Баланс клиента")
plt.title("Парная линейная регрессия: age и balance")
plt.legend()
plt.grid()
plt.show()


features = [
   "credit_score",
   "age",
   "tenure",
   "products_number",
   "credit_card",
   "active_member",
   "estimated_salary"
]
X = df[features]
y = df["balance"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


multi_model = LinearRegression()
multi_model.fit(X_train, y_train)


y_pred_multi = multi_model.predict(X_test)


print("Множественная линейная регрессия")
print("Коэффициенты:", multi_model.coef_)
print("Свободный член:", multi_model.intercept_)


mae_multi, mse_multi, rmse_multi, r2_multi = calculate_metrics(y_test, y_pred_multi)
print("MAE:", mae_multi)
print("MSE:", mse_multi)
print("RMSE:", rmse_multi)
print("R2:", r2_multi)


plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_multi, alpha=0.6)
plt.plot(
   [y_test.min(), y_test.max()],
   [y_test.min(), y_test.max()],
   "r--", label="Идеальный прогноз" )
plt.xlabel("Реальные значения balance")
plt.ylabel("Предсказанные значения balance")
plt.title("Множественная линейная регрессия")
plt.legend()
plt.grid()
plt.show()


tree_model = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)


y_pred_tree = tree_model.predict(X_test)


mae_tree, mse_tree, rmse_tree, r2_tree = calculate_metrics(y_test, y_pred_tree)
print("Дерево решений")
print("MAE:", mae_tree)
print("MSE:", mse_tree)
print("RMSE:", rmse_tree)
print("R2:", r2_tree)


plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_tree, alpha=0.6)
plt.plot(
   [y_test.min(), y_test.max()],
   [y_test.min(), y_test.max()],
   "r--", label="Идеальный прогноз" )
plt.xlabel("Реальные значения balance")
plt.ylabel("Предсказанные значения balance")
plt.title("Дерево решений")
plt.legend()
plt.grid()
plt.show()


boost_model = GradientBoostingRegressor(
   n_estimators=100,
   learning_rate=0.1,
   max_depth=3,
   random_state=42
)

boost_model.fit(X_train, y_train)


y_pred_boost = boost_model.predict(X_test)


mae_boost, mse_boost, rmse_boost, r2_boost = calculate_metrics(y_test, y_pred_boost)
print("Градиентный бустинг")
print("MAE:", mae_boost)
print("MSE:", mse_boost)
print("RMSE:", rmse_boost)
print("R2:", r2_boost)


plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_boost, alpha=0.6)
plt.plot(
   [y_test.min(), y_test.max()],
   [y_test.min(), y_test.max()],
   "r--", label="Идеальный прогноз"
)
plt.xlabel("Реальные значения balance")
plt.ylabel("Предсказанные значения balance")
plt.title("Градиентный бустинг")
plt.legend()
plt.grid()
plt.show()