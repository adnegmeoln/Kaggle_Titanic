import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

# 读取数据
train_df = pd.read_csv('D:/Projects/PycharmProjects/Poster/Data/train.csv')
test_df = pd.read_csv('D:/Projects/PycharmProjects/Poster/Data/test.csv')

# 去除Ticket和Cabin列
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

# 利用正则表达式获取乘客姓名中的Title
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)', expand=False)

# 把不常见的Title换成Rare，把一些Title转换为常见的形式
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                                 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# 把Title转换为序数的形式
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# 删除train数据中的Name和PassengerId列，删除test数据中的Name列
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

# 将性别转换为数字
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

# 补全年龄的缺失值，建立空数组
guess_ages = np.zeros((2, 3))

# 组合Pclass、Sex和Age数据
# 例如Pclass = 1，Sex = 0为一组，Pclass = 1，Sex = 1为一组，以此类推
# 取每组数据中年龄的中位数作为猜测年龄，补全年龄的缺失值
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # 将年龄中的浮点数转换为最近的0.5的倍数
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1),
                        'Age'] = guess_ages[i, j]

    dataset['Age'] = dataset['Age'].astype(int)

# 将Age依据年龄的区间转换为序数类型
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

combine = [train_df, test_df]

# 创建新特征FamilySize，从而可以去掉Parch和SibSp列
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# 创建另一个特征，叫做IsAlone
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# 去除Parch，SibSp和FamilySize列
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

# 构造一个特征Age*Class
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

# 训练数据中，Embarked有两个缺失值，用最常见的Embarked值填充
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# 将Embarked值转换为序数
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# 测试数据中有一个Fare值缺失，用众数填充
test_df['Fare'].fillna(test_df['Fare'].dropna().mode()[0], inplace=True)

# 把Fare值依据区间划分为序数
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

combine = [train_df, test_df]

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()

# Support Vector Machines
svc = SVC(kernel='poly', degree=4, gamma='auto')
svc.fit(X_train, Y_train)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
# Y_pred = svc.predict(X_test)

# Logistic Regression
logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
logreg.fit(X_train, Y_train)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
# Y_pred = logreg.predict(X_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
# Y_pred = knn.predict(X_test)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
# Y_pred = dtc.predict(X_test)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=200)
random_forest.fit(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
# Y_pred = random_forest.predict(X_test)

# Voting
voting = VotingClassifier(estimators=[('svc', svc), ('rf', random_forest), ('knn', knn), ('lr', logreg)], voting='hard')
voting = voting.fit(X_train, Y_train)
acc_voting = round(voting.score(X_train, Y_train) * 100, 2)
Y_pred = voting.predict(X_test)

models = pd.DataFrame({
    'Model': ['SVM', 'kNN', 'Random Forest', 'Decision Tree', 'Logistic Regression', 'Voting'],
    'Score': [acc_svc, acc_knn, acc_random_forest, acc_decision_tree, acc_log, acc_voting]
})
print(models.sort_values(by='Score', ascending=False))
print()

for clf, label in zip([svc, random_forest, knn, logreg, voting],
                      ['SVM', 'Random Forest', 'KNN', 'Logistic Regression', 'ensemble']):
    scores = cross_val_score(clf, X_train, Y_train, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# submission = pd.DataFrame({
#         "PassengerId": test_df["PassengerId"],
#         "Survived": Y_pred
# })
# submission.to_csv('D:/Projects/PycharmProjects/Poster/Submission/submission_voting.csv', index=False)
