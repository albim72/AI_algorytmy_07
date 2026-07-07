import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==================================================
# 1. Wczytanie danych
# ==================================================

data = load_breast_cancer()

X = data.data
y = data.target

print("Wymiary danych:", X.shape)
print("Klasy:", data.target_names)


# ==================================================
# 2. Podział danych
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ==================================================
# 3. Pipeline
# ==================================================

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=42
        )
    )
])


# ==================================================
# 4. Trening
# ==================================================

model.fit(X_train, y_train)


# ==================================================
# 5. Predykcja
# ==================================================

predicted = model.predict(X_test)

probabilities = model.predict_proba(
    X_test
)[:, 1]


# ==================================================
# 6. Ocena modelu
# ==================================================

accuracy = accuracy_score(
    y_test,
    predicted
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

print(f"\nAccuracy: {accuracy:.2%}")
print(f"ROC AUC: {roc_auc:.3f}")

print(
    classification_report(
        y_test,
        predicted,
        target_names=data.target_names
    )
)


# ==================================================
# 7. Macierz pomyłek
# ==================================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predicted,
    display_labels=data.target_names
)

plt.title("Macierz pomyłek: regresja logistyczna")
plt.show()


# ==================================================
# 8. Krzywa ROC
# ==================================================

RocCurveDisplay.from_predictions(
    y_test,
    probabilities
)

plt.title("Krzywa ROC: regresja logistyczna")
plt.show()


# ==================================================
# 9. Współczynniki modelu
# ==================================================

classifier = model.named_steps["classifier"]

print(
    "Liczba współczynników:",
    classifier.coef_.shape
)

print(
    "Wyraz wolny:",
    classifier.intercept_
)
