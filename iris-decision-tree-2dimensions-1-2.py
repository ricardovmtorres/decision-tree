import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
iris = pd.DataFrame(data.data)
# print(iris.head(7))

iris.columns = load_iris().feature_names
iris['target'] = data.target
# print(iris)
# print(iris.target.value_counts())

# trabalhando com apenas 2 dimensões

# simplificando a base, filtrando apenas os valores 1 e 2
# iris01 = iris[iris.target.isin([1,2])]
# print(iris01)

# filtrando apenas comprimento e tamanho da pétala, com '.loc'
iris01 = iris.loc[iris.target.isin([1,2]),['petal length (cm)', 'petal width (cm)', 'target']]
# print(iris01)
X = iris01.drop("target",axis=1)
y = iris01["target"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state= 42)

# utilizando o matplot para visualizar os dados de treino
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('_mpl-gallery')

# plot
fig, ax = plt.subplots()

# mapeando as classes para cores
class_to_color = {
  1: 'red',
  2: 'green'
}
# convertendo os rótulos de classe em cores
colors_train = [class_to_color[label] for label in y_train]

ax.scatter(X_train['petal length (cm)'], 
           X_train['petal width (cm)'], 
           c=colors_train)

# plt.show()


# classificação dos dados de treino e usando arvore de decisão
from sklearn import tree
#criando classificador
clf = tree.DecisionTreeClassifier(random_state=42)
#fazendo fit com os dados de treino
clf = clf.fit(X_train, y_train)
# verificando o score
print(clf.score(X_train, y_train))
# visualizando a arvore
fig, ax = plt.subplots(figsize=(10,8))
tree.plot_tree(clf)
plt.show()


# plot com a classificação
fig, ax = plt.subplots()

ax.scatter(X_train['petal length (cm)'], 
           X_train['petal width (cm)'], 
           c=colors_train)

ax.plot([5.05,5.05],[0.9, 2.7], '--r')
ax.plot([2.9,5.05],[1.9, 1.9], '--r')
ax.plot([2.9,5.05],[1.65, 1.65], '--r')
ax.plot([4.65,4.65],[1.65, 1.9], '--r')

# configurando a exibição do grafico
ax.set(xlim=(3, 7), xticks=[3,4,5,6,7],
       ylim=(0.9, 2.7), yticks=[1, 1.5, 2, 2.5])

plt.show()



# previsão
y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# plot com a previsão
fig, ax = plt.subplots()
colors_test = [class_to_color[label] for label in y_test]
ax.scatter(X_test['petal length (cm)'], 
           X_test['petal width (cm)'], 
           c=colors_test)

ax.plot([5.05,5.05],[0.9, 2.7], '--r')
ax.plot([2.9,5.05],[1.9, 1.9], '--r')
ax.plot([2.9,5.05],[1.65, 1.65], '--r')
ax.plot([4.65,4.65],[1.65, 1.9], '--r')

# configurando a exibição do grafico
ax.set(xlim=(3, 7), xticks=[3,4,5,6,7],
       ylim=(0.9, 2.7), yticks=[1, 1.5, 2, 2.5])

plt.show()