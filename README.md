# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Takes students' marks.

2.Trains a decision tree.

3.Finds rules like:

4."If Maths > 40 and Science > 50 → Pass"

Shows these rules as a tree diagram.5

## Program:

/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: 
RegisterNumber:  
*/
```
#  Import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Create simple dataset
data = {
    "Maths": [35, 78, 90, 45, 20, 60, 55, 30],
    "Science": [40, 85, 88, 50, 25, 65, 58, 35],
    "English": [45, 80, 92, 48, 30, 70, 60, 38],
    "Result": ["Pass", "Pass", "Pass", "Pass", "Fail", "Pass", "Pass", "Fail"]
}

df = pd.DataFrame(data)

X = df[["Maths", "Science", "English"]]
y = df["Result"]

#  Train Decision Tree
model = DecisionTreeClassifier(criterion="gini", max_depth=3)
model.fit(X, y)

#  Plot Decision Tree
plt.figure(figsize=(14, 8))
plot_tree(
    model,
    feature_names=["Maths", "Science", "English"],
    class_names=["Fail", "Pass"],
    filled=True
)
plt.title("Decision Tree: Student Result (Pass/Fail)")
plt.show()
```

## Output:
![decision tree classifier model](sam.png)
<img width="974" height="572" alt="Screenshot 2026-02-11 113943" src="https://github.com/user-attachments/assets/3230afba-fc4a-4f07-a93f-6c85c989795b" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
