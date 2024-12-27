# (c) 2024

import numpy as np
from scipy import optimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def choose_weights(X_quant,y_quant):
	X_quant_shape_0= X_quant.shape[0]
	X_quant_shape_1= X_quant.shape[1]
	print("X_quant_shape_0:",X_quant_shape_0)
	print("X_quant_shape_1:",X_quant_shape_1)
	def func(z):
		#matrix= X_quant
		#for k in range(X_quant_shape_1):
		#	for m in range(X_quant_shape_0):
		#		matrix[m][k]= matrix[m][k] * z[k]
		vector= np.zeros((X_quant_shape_0,1))
		for k in range(X_quant_shape_0):
			# print("k=",k)
			sum= 0
			for m in range(X_quant_shape_1):
				# print("k=",k," m=",m," X_quant[k][m]=",X_quant[k][m])
				A= X_quant[k][m] * z[m]
				# print("A=",A," np.log(A+1)=",np.log(A+1))
				L= np.log(A+1)
				# print("A=",A," np.log(A+1)=",L)
				sum= sum + L
				# sum= sum + A
			vector[k][0]= sum
		# print("vector=",vector)
		classifier= LogisticRegression(max_iter=10_000)
		# classifier.fit(matrix,y_quant)
		classifier.fit(vector,y_quant)
		# y_pred= classifier.predict(matrix)
		y_pred= classifier.predict(vector)
		accuracy= accuracy_score(y_quant,y_pred)
		# print(z,"->",accuracy)
		return -accuracy
	# Ranges= [ (-1.0, 1.0) for i in range(X_quant_shape_1)]
	Ranges= [ (0.0, 1.0) for i in range(X_quant_shape_1)]
	# print("Ranges:",Ranges)
	O= optimize.direct(func,Ranges)
	# O= optimize.direct(func,Ranges,maxfun=3000)
	# O= optimize.direct(func,Ranges,maxfun=10)
	# print("O:",O)
	# print("O.x:",O.x)
	return O.x
