from sklearn import tree

clf = tree.DecisionTreeClassifier()


# [height, weight, size boots]
X = [[181, 80, 42], [177, 70, 39], [160, 60, 36], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

clf = clf.fit(X, Y)

prediction = clf.predict([[160, 65, 30]])


print(prediction)