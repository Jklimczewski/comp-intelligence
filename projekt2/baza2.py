import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydot
from sklearn.model_selection import train_test_split
from sklearn import tree, metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

df2 = pd.read_csv("C:/infa/zajecia/4 semestr/io/projekt2/athletes.csv")

df2 = df2.dropna(thresh=14)
df2 = df2[(df2['gender'] == 'Male') | (df2['gender'] == 'Female')]
df2.drop(['athlete_id', 'eat', 'train', 'background', 'experience', 'schedule', 'howlong', 'fran', 'helen', 'grace',	'filthy50',
         'fgonebad', 'run400', 'run5k', 'name', 'region', 'team', 'affiliate'], axis=1, inplace=True)

mean_age = df2['age'].mean()
mean_weight = df2['weight'].mean()
mean_height = df2['height'].mean()
mean_candj = df2['candj'].mean()
mean_snatch = df2['snatch'].mean()
mean_deadlift = df2['deadlift'].mean()
mean_backsq = df2['backsq'].mean()
mean_pullups = df2['pullups'].mean()

df2['age'] = df2['age'].fillna(mean_age)
df2['weight'] = df2['weight'].fillna(mean_weight)
df2['height'] = df2['height'].fillna(mean_height)
df2['candj'] = df2['candj'].fillna(mean_candj)
df2['snatch'] = df2['snatch'].fillna(mean_snatch)
df2['deadlift'] = df2['deadlift'].fillna(mean_deadlift)
df2['backsq'] = df2['backsq'].fillna(mean_backsq)
df2['pullups'] = df2['pullups'].fillna(mean_pullups)


(train_set, test_set) = train_test_split(df2.values, train_size=0.7)
train_inputs = np.delete(train_set, 0, axis=1)
train_classes = train_set[:, 0]
test_inputs = np.delete(test_set, 0, axis=1)
test_classes = test_set[:, 0]

dtc = tree.DecisionTreeClassifier()
dtc2 = tree.DecisionTreeClassifier(max_depth=2)
naive_bay = GaussianNB()
knn3 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn5 = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn11 = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
mlp = MLPClassifier(hidden_layer_sizes=(2), activation='relu', max_iter=500, validation_fraction=0.1, early_stopping=True)
mlp2 = MLPClassifier(hidden_layer_sizes=(3), activation='tanh', max_iter=500, validation_fraction=0.1, early_stopping=True)
mlp3 = MLPClassifier(hidden_layer_sizes=(6, 3), activation='logistic', max_iter=500, validation_fraction=0.1, early_stopping=True)

dtc.fit(train_inputs, train_classes)
dtc2.fit(train_inputs, train_classes)
naive_bay.fit(train_inputs, train_classes)
knn3.fit(train_inputs, train_classes)
knn5.fit(train_inputs, train_classes)
knn11.fit(train_inputs, train_classes)
mlp.fit(train_inputs, train_classes)
mlp2.fit(train_inputs, train_classes)
mlp3.fit(train_inputs, train_classes)

loss_curve = mlp.loss_curve_
val_loss = mlp.validation_scores_
loss_curve2 = mlp2.loss_curve_
val_loss2 = mlp2.validation_scores_
loss_curve3 = mlp3.loss_curve_
val_loss3 = mlp3.validation_scores_

plt.plot(loss_curve)
plt.plot(val_loss)
plt.title('Loss Curve1')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

plt.plot(loss_curve2)
plt.plot(val_loss2)
plt.title('Loss Curve2')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

plt.plot(loss_curve3)
plt.plot(val_loss3)
plt.title('Loss Curve3')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

dtc_accuracy = dtc.score(test_inputs, test_classes)
dtc2_accuracy = dtc2.score(test_inputs, test_classes)
naive_bay_accuracy = naive_bay.score(test_inputs, test_classes)
knn3_accuracy = knn3.score(test_inputs, test_classes)
knn5_accuracy = knn5.score(test_inputs, test_classes)
knn11_accuracy = knn11.score(test_inputs, test_classes)
mlp_accuracy = mlp.score(test_inputs, test_classes)
mlp2_accuracy = mlp2.score(test_inputs, test_classes)
mlp3_accuracy = mlp3.score(test_inputs, test_classes)
print("Procent poprawności dla DTC w większej wersji drzewa: ", dtc_accuracy, "\n")
print("Procent poprawności dla DTC w mniejsze wersji drzewa: ", dtc2_accuracy, "\n")
print("Procent poprawności dla Naive Bayes: ", naive_bay_accuracy, "\n")
print("Procent poprawności dla KNN przy k=3: ", knn3_accuracy, "\n")
print("Procent poprawności dla KNN przy k=5: ", knn5_accuracy, "\n")
print("Procent poprawności dla KNN przy k=11: ", knn11_accuracy, "\n")
print("Procent poprawności dla MLP przy 2 neuronach w warstwie: ", mlp_accuracy, "\n")
print("Procent poprawności dla MLP przy 3 neuronach w warstwie: ", mlp2_accuracy, "\n")
print("Procent poprawności dla MLP przy 2 warstwach z (6, 3) neuronami: ", mlp3_accuracy, "\n")

dtc_y_pred = dtc.predict(test_inputs)
dtc2_y_pred = dtc2.predict(test_inputs)
naive_bay_y_pred = naive_bay.predict(test_inputs)
knn3_y_pred = knn3.predict(test_inputs)
knn5_y_pred = knn5.predict(test_inputs)
knn11_y_pred = knn11.predict(test_inputs)
mlp_y_pred = mlp.predict(test_inputs)
mlp2_y_pred = mlp2.predict(test_inputs)
mlp3_y_pred = mlp3.predict(test_inputs)
print("Macierz błędów dla DTC w większej wersji drzewa: \n", metrics.confusion_matrix(test_classes, dtc_y_pred), "\n")
print("Macierz błędów dla DTC w mniejsze wersji drzewa: \n", metrics.confusion_matrix(test_classes, dtc2_y_pred), "\n")
print("Macierz błędów dla Naive Bayes: \n", metrics.confusion_matrix(test_classes, naive_bay_y_pred), "\n")
print("Macierz błędów dla KNN przy k=3: \n", metrics.confusion_matrix(test_classes, knn3_y_pred), "\n")
print("Macierz błędów dla KNN przy k=5: \n", metrics.confusion_matrix(test_classes, knn5_y_pred), "\n")
print("Macierz błędów dla KNN przy k=11: \n", metrics.confusion_matrix(test_classes, knn11_y_pred), "\n")
print("Macierz błędów dla MLP przy 2 neuronach w warstwie: \n", metrics.confusion_matrix(test_classes, mlp_y_pred), "\n")
print("Macierz błędow dla MLP przy 3 neuronach w warstwie: \n", metrics.confusion_matrix(test_classes, mlp2_y_pred), "\n")
print("Macierz błędów dla MLP przy 2 warstwach z (6, 3) neuronami: \n", metrics.confusion_matrix(test_classes, mlp3_y_pred), "\n")