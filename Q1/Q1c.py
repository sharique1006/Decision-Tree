import pandas as pd
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def getData(file):
	data = pd.read_csv(file, index_col=None)
	x = data.iloc[:,:-1].to_numpy()
	y = data.iloc[:,-1].to_numpy()
	return x, y

def bestParameters(x, y):
	n_estimators = [50, 150, 250, 350, 450]
	max_features = [0.1, 0.3, 0.5, 0.7, 0.9]
	min_samples_split = [2, 4, 6, 8, 10]
	param_grid = [{'n_estimators': n_estimators, 'max_features': max_features, 'min_samples_split': min_samples_split}]
	clf = GridSearchCV(RandomForestClassifier(), param_grid, n_jobs=-1, verbose=10)
	start = time.time()
	clf.fit(x, y)
	end = time.time()
	print("Tuning Time = ", (end-start))
	best_params = clf.best_params_
	print("Result = ", clf.cv_results_)
	print("Best Parameters = ", best_params)
	return best_params

def RFC(best_params, x_train, y_train, x_val, y_val, x_test, y_test):
	n = best_params['n_estimators']
	f = best_params['max_features']
	s = best_params['min_samples_split']

	clf = RandomForestClassifier(n_estimators=n, max_features=f, min_samples_split=s, oob_score=True)
	start = time.time()
	clf.fit(x_train, y_train)
	end = time.time()
	print("Training Time = ", (end-start))
	train_acc = clf.score(x_train, y_train)
	val_acc = clf.score(x_val, y_val)
	test_acc = clf.score(x_test, y_test)
	oob_score = clf.oob_score_
	print("Training Accuracy = ", train_acc)
	print("Validation Accuracy = ", val_acc)
	print("Test Accuracy = ", test_acc)
	print("Out of Bag Accuracy = ", oob_score)

def plotAccuracyvsParameters(best_params, x_train, y_train, x_val, y_val, x_test, y_test):
	n = best_params['n_estimators']
	f = best_params['max_features']
	s = best_params['min_samples_split']
	n_estimators = [50, 150, 250, 350, 450]
	max_features = [0.1, 0.3, 0.5, 0.7, 0.9]
	min_samples_split = [2, 4, 6, 8, 10]

	val_acc1 = []
	test_acc1 = []
	for x in min_samples_split:
		clf = RandomForestClassifier(n_estimators=n, max_features=f, min_samples_split=x)
		clf.fit(x_train, y_train)
		val_acc1.append(clf.score(x_val, y_val))
		test_acc1.append(clf.score(x_test, y_test))

	print("n_estimators = ", n, "max_features = ", f, "min_samples_split = ", min_samples_split)
	print("Val Accuracy = ", val_acc1)
	print("Test Accuracy = ", test_acc1)

	plt.figure()
	plt.plot(min_samples_split, val_acc1, label='Validation Accuracy')
	plt.plot(min_samples_split, test_acc1, label='Test Accuracy')
	plt.title('Accuracy vs Min Samples Split')
	plt.xlabel('Min Samples Split')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig('min_samples.png')
	plt.show()
	plt.close()

	val_acc2 = []
	test_acc2 = []
	for x in max_features:
		clf = RandomForestClassifier(n_estimators=n, max_features=x, min_samples_split=s)
		clf.fit(x_train, y_train)
		val_acc2.append(clf.score(x_val, y_val))
		test_acc2.append(clf.score(x_test, y_test))

	print("n_estimators = ", n, "max_features = ", max_features, "min_samples_split = ", s)
	print("Val Accuracy = ", val_acc2)
	print("Test Accuracy = ", test_acc2)

	plt.figure()
	plt.plot(max_features, val_acc2, label='Validation Accuracy')
	plt.plot(max_features, test_acc2, label='Test Accuracy')
	plt.title('Accuracy vs Max Features')
	plt.xlabel('Max Features')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig('max_features.png')
	plt.show()
	plt.close()

	val_acc3 = []
	test_acc3 = []
	for x in n_estimators:
		clf = RandomForestClassifier(n_estimators=x, max_features=f, min_samples_split=s)
		clf.fit(x_train, y_train)
		val_acc3.append(clf.score(x_val, y_val))
		test_acc3.append(clf.score(x_test, y_test))

	print("n_estimators = ", n_estimators, "max_features = ", f, "min_samples_split = ", s)
	print("Val Accuracy = ", val_acc3)
	print("Test Accuracy = ", test_acc3)

	plt.figure()
	plt.plot(n_estimators, val_acc3, label='Validation Accuracy')
	plt.plot(n_estimators, test_acc3, label='Test Accuracy')
	plt.title('Accuracy vs Number of Estimators')
	plt.xlabel('Number of Estimators')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.savefig('n_estimators.png')
	plt.show()
	plt.close()

x_train, y_train = getData('../decision_tree/decision_tree/train.csv')
x_val, y_val = getData('../decision_tree/decision_tree/val.csv')
x_test, y_test = getData('../decision_tree/decision_tree/test.csv')

best_params = bestParameters(x_train, y_train)
RFC(best_params, x_train, y_train, x_val, y_val, x_test, y_test)
plotAccuracyvsParameters(best_params, x_train, y_train, x_val, y_val, x_test, y_test)
