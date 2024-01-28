import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import warnings

dataset = pd.read_csv("nba2021.csv")
X = dataset.drop(['Player', 'Pos', 'Tm', 'GS', 'FG%','2P%','3P%','FTA','STL','TOV'], axis=1)
Y = dataset['Pos']

#Split
x_train, x_test, y_train, y_test = train_test_split(X,Y, stratify=dataset['Pos'], random_state=0, train_size=0.75, test_size=0.25)

#standardization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
X = scaler.fit_transform(X)

model = svm.SVC(kernel='linear',C=8)
model.fit(x_train,y_train)
training_accuracy = model.score(x_train,y_train)
test_accuracy = model.score(x_test, y_test)

print("-----------------------------Task 1-----------------------------------")
print("Training accuracy:{:.3f}, Test accuracy:{:.3f} \n".format(training_accuracy, test_accuracy))

prediction = model.predict(x_test)
print("-----------------------------Task 2------------------------------------")
print("Confusion matrix:")
print(pd.crosstab(y_test, prediction, rownames=['True'], colnames=['Predicted'], margins=True),"\n")
warnings.filterwarnings("ignore")

print("-----------------------------Task 3------------------------------------")
scores = cross_val_score(model, X, Y, cv=10)
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))
warnings.filterwarnings("ignore")