from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# Para Random Forest:
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_score, f1_score, confusion_matrix


def transformarcalidad(point):
    if point < 6:
        return 0
    elif point < 8:
        return 1
    else:
        return 2


def barplot(data, labels, path_to_save):
    heights = []

    for i in labels:
        heights.append(0)

    for i in range(len(labels)):
        for x in data:
            if x == i:
                heights[i] += 1

    plt.figure()
    plt.bar([i for i in range(len(labels))], heights, tick_label=labels)
    plt.show()
    plt.waitforbuttonpress()


# Cargar datos
data = pd.read_csv("winequality-red.csv", delimiter=";")

#
# data = data.drop(["date"], axis=1)

data["quality"] = data["quality"].transform(transformarcalidad)

# Visualizamos un poco los datos primero
for key, value in data.iteritems():
    print(key, value.describe())
    print()

# Habria que normalizar
#barplot(data["quality"].to_numpy(), ["0", "1", "2"], None)

X = data[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
          "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]
Y = data[["quality"]]

# Split en datos de entrenamiento y test
# Es importante separar los datos porque necesitamos algunos datos en los que
# no se haya aprendido para sacar conclusiones correctas
# (Se trata de emular datos "nuevos", pero en los que sabemos el resultado
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)


clf = BaggingClassifier(base_estimator=LogisticRegression(max_iter=1000, class_weight={0: 0.2, 1: 0.2, 2: 1}), n_estimators=20)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("\t\t\t\t0\t1\t2")
print("0\t\t\t\t{}\t{}\t{}".format(cm[0, 0], cm[0, 1], cm[0, 2]))
print("1\t\t\t\t{}\t{}\t{}".format(cm[1, 0], cm[1, 1], cm[1, 2]))
print("2\t\t\t\t{}\t{}\t{}".format(cm[2, 0], cm[2, 1], cm[2, 2]))

