import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

def getData(file):
	data = pd.read_csv(file, index_col=None)
	x = data.iloc[:,:-1].to_numpy()
	y = data.iloc[:,-1].to_numpy()
	return x, y

node_index = 0

class Node():
	def __init__(self, attribute, split_value, label, parent):
		global node_index
		self.attribute = attribute
		self.split_value = split_value
		self.label = label
		self.parent = parent
		self.right = None
		self.left = None
		self.node_number = node_index
		node_index += 1

class DecisionTree():
	def __init__(self):
		self.root = None
		self.num_nodes = None

	def CBA(self, x, y):
		MI = []
		labels, counts = np.unique(y, return_counts=True)
		P_y = [(y == l).sum()/x.shape[0] for l in labels]
		H_y =  -1 * (P_y * np.log(P_y)).sum()
		for col in range(x.shape[1]):
			attribute = (x[:,col] >= np.median(x[:,col])).astype(np.int64)
			unique_attr = np.unique(attribute)
			if unique_attr.shape[0] == 1:
				MI.append(-1)
				continue
			P_xj = [(attribute == xj).sum()/x.shape[0] for xj in unique_attr]
			H_y_xj = np.zeros(unique_attr.shape)
			for idx, xj in enumerate(unique_attr):
				P_y_xj = np.zeros(labels.shape)
				for idy, l in enumerate(labels):
					P_y_xj[idy] = ((attribute == xj)*(y == l)).sum()/(attribute == xj).sum()
				H_y_xj[idx] = -1 * (P_y_xj * np.log(P_y_xj + (P_y_xj == 0))).sum()
			MI_col = H_y - (P_xj*H_y_xj).sum()
			MI.append(MI_col)
		if (np.array(MI) == -1).all():
			return None, None
		split_attribute = np.argmax(MI)
		xj = np.unique((x[:,split_attribute] >= np.median(x[:,split_attribute])).astype(np.int64))
		return split_attribute, xj, labels[np.argmax(counts)]

	def growTree(self, x, y):
		def growTreeRec(x, y, node, parent):
			if (y == y[0]).all():
				return Node(None, None, y[0], node)
			col, xj, maxlabel = self.CBA(x, y)
			if xj is None:
				label, counts = np.unique(y, return_counts=True)
				maxcount_label = label[np.argmax(counts)]
				return Node(None, None, maxcount_label, node)
			attr = (x[:,col] >= np.median(x[:,col])).astype(np.int64)
			idx = [np.where(attr == k) for k in xj]
			Sx0, Sy0 = x[idx[0]], y[idx[0]]
			Sx1, Sy1 = x[idx[1]], y[idx[1]]

			node = Node(col, np.median(x[:,col]), maxlabel, parent)
			node.left = growTreeRec(Sx0, Sy0, node.left, node)
			node.right = growTreeRec(Sx1, Sy1, node.right, node)
			return node
		self.root = growTreeRec(x, y, self.root, None) 
		self.num_nodes = node_index

	def predict(self, x, y):
		def predictRec(x, node):
			if node.attribute == None:
				return node.label
			if x[node.attribute] < node.split_value:
				if node.left != None: return predictRec(x, node.left)
			if node.right != None: return predictRec(x, node.right)
			return node.label

		prediction = []
		for i in range(len(x)):
			prediction.append(predictRec(x[i], self.root))

		accuracy = (y == prediction).sum()/len(y)
		return np.array(prediction), accuracy

	def height(self, node):
		if node.left == None and node.right == None:
			return 0
		return max(self.height(node.left), self.height(node.right)) + 1