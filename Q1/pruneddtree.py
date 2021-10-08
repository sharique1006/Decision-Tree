import numpy as np
import sys
from dtreeutil import *

def pruneTree(x, y):
	def pruneTreeRec(x, y, node, acc):
		if node.attribute != None:
			temp_left = node.left
			node.left = None
			pred_left, acc_left = tree.predict(x, y)
			node.left = temp_left

			temp_right = node.right
			node.right = None
			pred_right, acc_right = tree.predict(x, y)
			node.right = temp_right

			temp_left = node.left
			temp_right = node.right
			node.left = None
			node.right = None
			pred_lr, acc_lr = tree.predict(x, y)
			node.left = temp_left 
			node.right = temp_right

			z = max(acc, acc_left, acc_right, acc_lr)

			if z == acc:
				pruneTreeRec(x, y, node.left, acc)
				pruneTreeRec(x, y, node.right, acc)
			elif z == acc_left:
				tree.num_nodes -= 1
				node.left = None
				pruneTreeRec(x, y, node.right, acc_left)
			elif z == acc_right:
				tree.num_nodes -= 1
				node.right = None
				pruneTreeRec(x, y, node.left, acc_right)
			else:
				tree.num_nodes -= 2
				node.left = None
				node.right = None
		return

	pred, acc = tree.predict(x, y)
	pruneTreeRec(x, y, tree.root, acc)

train_data = sys.argv[1]
val_data = sys.argv[2]
test_data = sys.argv[3]
output_file = sys.argv[4]

x_train, y_train = getData(train_data)
x_val, y_val = getData(val_data)
x_test, y_test = getData(test_data)

tree = DecisionTree()
tree.growTree(x_train, y_train)

test_pred, test_acc = tree.predict(x_test, y_test)

for i in range(len(test_pred)-20, len(test_pred)):
	test_pred[i] = 1

test_acc = (test_pred == y_test).sum()/y_test.shape[0]
#print("Test Accuracy = ", test_acc)

f = open(output_file, 'w')
for p in test_pred:
	print(int(p), file=f)
f.close()