'''
I was going to do RandomTreesEmbedding becuase the scikit documentation specifically mentioned that it can handle high dimensions well *
and this is a classification task for a breast cancer dataset with 30 features. It's a high-ish dimension.
But I later realized that it's an unsupervised transformer, that seemed like a lot of work to get it working so
I pivoted to DecisionTreeClassifier. Also why is RandomTreesEmbedding under supervised learning in the docs?
Anyway, as the name suggests DecisionTreeClassifier is an older, superior sibling to RandomTreesEmbedding.
That's oversimplifing, but the gist is that the DecisionTreeClassifier is a supervised classifier, 
it uses labels to train trees and limit classification error.
But RandomTreesEmbedding is an unsupervised transformer (not a classifier), it ignores labels, uses completely random tree splits 
and requires another classifier for the classifications.
In other words, DecisionTreeClassifier uses the same tree structure and is easier to implement.

* I was wrong.
The docs doesn't say RandomTreesEmbedding can handle high dimensional inputs well, it was talking about high dimensional outputs.
The documentation was quite technical:
"Using a forest of completely random trees, RandomTreesEmbedding encodes the data by the indices of the leaves a data point ends up in. 
This index is then encoded in a one-of-K manner, leading to a high dimensional, sparse binary coding." - 1.11.2.6. Totally Random Trees Embedding.
I've already implemented DecisionTreeClassifier and it worked, so . . . .
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from binary_classification import load_data, train, predict, accuracy


#from-scratch
X_train, X_test, y_train, y_test, trash = load_data()
w, b, losses = train(X_train, y_train)
prediction = predict(X_test, w, b)
accuracies = accuracy(y_test, prediction)

#decision tree classfier
tree = DecisionTreeClassifier(random_state=17) #Ohtani
tree.fit(X_train.numpy(), y_train.numpy())
tree_prediction = tree.predict(X_test.numpy())
tree_accuracy = accuracy_score(y_test.numpy(), tree_prediction)

print(f'from-scratch accuracy: {accuracies}')
print(f'decision tree classfier accuracy: {tree_accuracy}')

'''
from-scratch accuracy: 0.9912280440330505
decision tree classfier accuracy: 0.9385964912280702

I think they may be overfitting, especially the from-scratch model.
The second accuracy of 94% is more believable, but scikit docs do warn that decision trees tend to overfit.
'''

