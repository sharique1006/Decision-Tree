import numpy as np
import sys
from dtreeutil import *

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

#print("Number of Nodes = ", tree.num_nodes)
#print("Test Accuracy = ", test_acc)

f = open(output_file, 'w')
for p in test_pred:
	print(int(p), file=f)
f.close()
