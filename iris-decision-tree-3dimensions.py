import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
iris = pd.DataFrame(data.data)

iris.columns = load_iris().feature_names
iris['target'] = data.target

X = iris.drop("target",axis=1)
y = iris["target"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state= 42)

# utilizando o matplot para visualizar os dados de treino
import matplotlib.pyplot as plt
plt.style.use('_mpl-gallery')

# mapeando as classes para cores
class_to_color = {
  0: 'red',
  1: 'green',
  2: 'blue'
}
# convertendo os rótulos de classe em cores
colors_train = [class_to_color[label] for label in y_train]

# classificação dos dados de treino e usando arvore de decisão
from sklearn import tree
#criando classificador
clf2 = tree.DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
# verificando o score
print(clf2.score(X_train, y_train))
# visualizando a arvore
fig, ax = plt.subplots(figsize=(10,8))
tree.plot_tree(clf2)
plt.show()

# previsão
y_pred2 = clf2.predict(X_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred2))


# plot com a previsão
# fig, ax = plt.subplots()
# colors_test = [class_to_color[label] for label in y_test]
# ax.scatter(X_test['petal length (cm)'], 
#            X_test['petal width (cm)'], 
#            c=colors_test)

# ax.plot([5.05,5.05],[0.9, 2.7], '--r')
# ax.plot([2.9,5.05],[1.9, 1.9], '--r')
# ax.plot([2.9,5.05],[1.65, 1.65], '--r')
# ax.plot([4.65,4.65],[1.65, 1.9], '--r')

# # configurando a exibição do grafico
# ax.set(xlim=(3, 7), xticks=[3,4,5,6,7],
#        ylim=(0.9, 2.7), yticks=[1, 1.5, 2, 2.5])

# plt.show()