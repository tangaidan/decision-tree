# decision-tree
from sklearn.datasets import load_iris
iris = load_iris()

#将数据分成训练集和测试集
from sklearn import cross_validation
x_train, x_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.3, random_state=0)
#random_state表示随机种子，为0时说明每次随机

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='gini')
#gini默认方式，即CART算法；entropy使用信息增益，即ID3、C4.5算法
clf.fit(x_train, y_train)
clf.predict(x_test)
# the probability of each class
clf.predict_proba(x_test)

#决策树可视化
from sklearn import tree
with open("iris1.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
