import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==================================================
# 1. Wczytanie danych
# ==================================================

iris = load_iris()

X = iris.data
y = iris.target

print("Wymiary danych:", X.shape)
print("Nazwy cech:", iris.feature_names)
print("Nazwy klas:", iris.target_names)


# ==================================================
# 2. Podział train-test
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ==================================================
# 3. Pipeline: skalowanie + k-NN
# ==================================================

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            metric="minkowski",
            p=2
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


# ==================================================
# 6. Ocena
# ==================================================

accuracy = accuracy_score(
    y_test,
    predicted
)

print(f"\nAccuracy: {accuracy:.2%}")

print(
    classification_report(
        y_test,
        predicted,
        target_names=iris.target_names
    )
)


# ==================================================
# 7. Macierz pomyłek
# ==================================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predicted,
    display_labels=iris.target_names
)

plt.title("Macierz pomyłek: k-NN")
plt.show()


# ==================================================
# 8. Predykcja nowego kwiatu
# ==================================================

new_flower = [[
    5.8,  # długość działki kielicha
    2.7,  # szerokość działki kielicha
    4.1,  # długość płatka
    1.0   # szerokość płatka
]]

new_prediction = model.predict(new_flower)
probabilities = model.predict_proba(new_flower)

print(
    "Przewidziany gatunek:",
    iris.target_names[new_prediction[0]]
)

print(
    "Prawdopodobieństwa:",
    probabilities[0]
)
