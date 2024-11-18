from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=500, n_features=10, n_informative=8, n_redundant=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

weak_learner = DecisionTreeClassifier(max_depth=1)

adaboost = AdaBoostClassifier(base_estimator=weak_learner, n_estimators=50, random_state=42)

adaboost.fit(X_train, y_train)
y_pred = adaboost.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of AdaBoost Classifier: {accuracy:.2f}")
