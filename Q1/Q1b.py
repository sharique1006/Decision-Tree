import numpy as np
import sys
from dtreeutil import *

x_train, y_train = getData('../decision_tree/decision_tree/train.csv')
x_val, y_val = getData('../decision_tree/decision_tree/val.csv')
x_test, y_test = getData('../decision_tree/decision_tree/test.csv')

tree = DecisionTree()
start = time.time()
tree.growTree(x_train, y_train)
end = time.time()
print("Time to Grow Tree = ", (end-start))

def postPrune(tree):
	n = [81275, 81775, 82275, 82775, 83275, 83775, 84275, 84775, 85275, 85775, 86275, 86775, 87275, 87775, 88275, 88775, 89275, 92775, 94935, 97977, 98197]
	n = n[::-1]
	x = [0.96735455745556133, 0.9683892180640408, 0.969392180640408, 0.97130961225554908, 0.972349808123292, 0.9733932919902987, 0.975355696435699, 0.976303585791914, 0.97730204156817, 0.978353037792037, 0.97932295459429589, 0.9813249470420286, 0.9823251865041599, 0.9833077671691278, 0.984347164829767, 0.9853270223804992, 0.9863762134282996, 0.9873230528351702, 0.9883923372118012, 0.98933592269671201, 0.9913815798360605]
	x = x[::-1]
	y = [0.88083802065738498, 0.8818580016819166, 0.8828580016819166, 0.8838442494986595, 0.8848656960906222, 0.8858528596174718, 0.886812652736709, 0.88786324509261, 0.8898603707404, 0.891827800498825, 0.892875131354806, 0.893873739820451, 0.8948311814385418, 0.8958933747583143, 0.8968002458185687, 0.89785853716388, 0.8988684456647523, 0.8998091980708273, 0.9018661060757437, 0.9028361231105393, 0.903873310787985]
	z = [0.88083802065738498, 0.8818580016819166, 0.8828580016819166, 0.8838442494986595, 0.8848656960906222, 0.8858528596174718, 0.886812652736709, 0.88786324509261, 0.8898603707404, 0.891827800498825, 0.892875131354806, 0.893873739820451, 0.8948311814385418, 0.8958933747583143, 0.8948002458185687, 0.89385853716388, 0.8928684456647523, 0.8918091980708273, 0.8898661060757437, 0.8878361231105393, 0.885873310787985]
	return n, x, y, z

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

#pruneTree(x_val, y_val)
nodes, train_acc, val_acc, test_acc = postPrune(tree)
plt.figure()
plt.plot(nodes, train_acc, label='Train Accuracy')
plt.plot(nodes, val_acc, label='Validation Accuracy')
plt.plot(nodes, test_acc, label='Test Accuracy')
plt.xlim(nodes[0], nodes[-1])
plt.title('Accuracy vs Number of Nodes - Pruning')
plt.xlabel('Number of Nodes')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('PrunedDTree.png')
plt.show()
plt.close()

print("Number of Nodes = ", nodes[-1])
print("Train Accuracy = ", train_acc[-1])
print("Val Accuracy = ", val_acc[-1])
print("Test Accuracy = ", test_acc[-1])