import numpy as np
from dtreeutil import *

x_train, y_train = getData('../decision_tree/decision_tree/train.csv')
x_val, y_val = getData('../decision_tree/decision_tree/val.csv')
x_test, y_test = getData('../decision_tree/decision_tree/test.csv')

tree = DecisionTree()
start = time.time()
tree.growTree(x_train, y_train)
end = time.time()
print("Time to Grow Tree = ", (end-start))

train_pred, train_acc = tree.predict(x_train, y_train)
test_pred, test_acc = tree.predict(x_test, y_test)
val_pred, val_acc = tree.predict(x_val, y_val)

print("Number of Nodes = ", tree.num_nodes)
print("Train Accuracy = ", train_acc)
print("Test Accuracy = ", test_acc)
print("Val Accuracy = ", val_acc)

def predictByLevel(x, y, node, stop_level, curr_level):
	if stop_level == curr_level:
		return node.label
	if node.attribute == None:
		return node.label
	if x[node.attribute] < node.split_value:
		return predictByLevel(x, y, node.left, stop_level, curr_level+1)
	return predictByLevel(x, y, node.right, stop_level, curr_level+1)

def predict(x, y):
	accuracies = []
	height = tree.height(tree.root)
	for i in range(height):
		prediction = []
		for j in range(x.shape[0]):
			prediction.append(predictByLevel(x[j], y[j], tree.root, i, 0))
		acc = (y == prediction).sum()/y.shape[0]
		accuracies.append(acc)
	return accuracies

count = 0
def traversal(node, stop_level, curr_level):
	global count
	count += 1
	if curr_level == stop_level:
		return
	if node.attribute != None:
		traversal(node.left, stop_level, curr_level+1)
		traversal(node.right, stop_level, curr_level+1)

train_acc = predict(x_train, y_train)
test_acc = predict(x_test, y_test)
val_acc = predict(x_val, y_val)

height = tree.height(tree.root)        
nodesAtDepth = []
for i in range(height):
	traversal(tree.root, i, 0)
	nodesAtDepth.append(count)
	count = 0

plt.figure()
plt.plot(nodesAtDepth, train_acc, label='Train Accuracy')
plt.plot(nodesAtDepth, test_acc, label='Test Accuracy')
plt.plot(nodesAtDepth, val_acc, label='Val Accuracy')
plt.title('Accuracy vs Number of Nodes')
plt.xlabel('Number of Nodes')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('DTree.png')
plt.show()
plt.close()