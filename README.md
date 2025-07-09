# iris-mini-classification
Description Statement:
Iris is a flower which consist of a wide range of species but majorly catogirised as iris-setosa,iris-versicolor,iris-virginica so we will use a machine learning model to make machine learn how to classify the flower based on its parameters or briefly stated sepal length,sepal width,petal length,petal width.
<pre>
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv('https://gist.github.com/Thanatoz-1/9e7fdfb8189f0cdf5d73a494e4a6392a')
x = df.drop('species', axis=1)  # assuming 'species' is your target column
y = df['species']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier()
model.fit(x_train,y_train)
ypred=model.predict(x_test)
print("accuracy:",accuracy_score(y_test,ypred))
</pre>
