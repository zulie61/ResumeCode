import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
#import numpy as np
#前處理
train = pd.read_csv("train.csv")
train["Age"] = train["Age"].fillna(int(train["Age"].median()))


train.loc[train["Sex"] == "male" ,'Sex'] = 0 
train.loc[train["Sex"] == "female" ,'Sex'] = 1 


train.loc[train["Embarked"] == "S" , "Embarked" ] = 0
train.loc[train["Embarked"] == "C" , "Embarked" ] = 1
train.loc[train["Embarked"] == "Q" , "Embarked" ] = 2
train["Embarked"] = train["Embarked"].fillna(int(train["Embarked"].median()))


train["Child"] = float('NaN')
train.loc[train["Age"] < 18 , "Child" ] = 1
train.loc[train["Age"] >= 18 , "Child" ] = 0
# 預測test.csv裡乘客生還與否，並根據上述結果，假設男性皆罹難，女性皆生還


test = pd.read_csv("test.csv")
test ["Survived"] = 0
test.loc[test ["Sex"] == "female" , "Survived"] = 1
test["Age"] = test["Age"].fillna(int(test["Age"].median()))


test.loc[test["Sex"] == "male" ,'Sex'] = 0 
test.loc[test["Sex"] == "female" ,'Sex'] = 1 


test.loc[test["Embarked"] == "S" , "Embarked" ] = 0
test.loc[test["Embarked"] == "C" , "Embarked" ] = 1
test.loc[test["Embarked"] == "Q" , "Embarked" ] = 2
test["Embarked"] = test["Embarked"].fillna(int(test["Embarked"].median()))
test["Fare"] = test["Fare"].fillna(int(test["Fare"].median()))

print("4特徵樹")
#train集訓練
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values
features_one = scale(features_one)#標準化=>提高分類的準確性
tree1 = tree.DecisionTreeClassifier(min_samples_leaf=5 ,min_samples_split=144, random_state = 1)
tree1 = tree1.fit(features_one, target)
#測試模型的有效性 0.9775533108866442
print("模型有效性:"+ str(tree1.score(features_one,target))) 
#特徵的重要性,質越大越重要
print("特徵重要性:"+ str(tree1.feature_importances_))

#test集訓練
features_test = test[["Pclass", "Sex", "Age", "Fare"]].values
features_test = scale(features_test)
my_prediction = tree1.predict(features_test)
#print(my_prediction)
#print(test["Survived"].values)
sucnum = 0
for i in range(len(my_prediction)):
    if  my_prediction[i] == test["Survived"].values[i] :
        sucnum +=1
print("準確次數" + str(sucnum))
print("準確率" + str(sucnum/len(my_prediction)))

print("------------------------------")

print("全特徵樹")
features_two = train[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values
features_two = scale(features_two)
tree2 = tree.DecisionTreeClassifier()
tree2 = tree2.fit(features_two,target)
print("模型有效性:"+ str(tree2.score(features_two,target)))
print("特徵重要性:"+ str(tree2.feature_importances_))
features2_test = test[["Pclass","Age","Sex","Fare", "SibSp", "Parch", "Embarked"]].values
features2_test = scale(features2_test)
my_prediction2 = tree2.predict(features2_test)
sucnum2 = 0
for i in range(len(my_prediction)):
    if  my_prediction2[i] == test["Survived"].values[i] :
        sucnum2 +=1
print("準確次數" + str(sucnum2))
print("準確率" + str(sucnum2/len(my_prediction2)))

print("------------------------------")

features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
forest = forest.fit(features_forest, target)
print("模型有效性:"+ str(forest.score(features_forest, target)))
print("特徵重要性:"+ str(forest.feature_importances_))
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
test_features = scale(test_features)
pred_forest = forest.predict(test_features)
sucnum3 = 0
for i in range(len(pred_forest)):
    if  pred_forest[i] == test["Survived"].values[i] :
        sucnum3 +=1
print("準確次數" + str(sucnum3))
print("準確率" + str(sucnum3/len(pred_forest)))

