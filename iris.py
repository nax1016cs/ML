import numpy as np
import pandas as pd
import math
import random

df =pd.read_csv("iris.data", header = None, names = ["sepal_length","sepal_width","petal_length", "petal_width", "name"])
df = df.sample(frac=1).reset_index(drop=True)

name = ["Iris-setosa","Iris-versicolor","Iris-virginica"]
feature = ["sepal_length","sepal_width","petal_length", "petal_width"]
# #
# n = ["mean", "std"]
# seta = pd.DataFrame(index=n, columns = feature)
# df_set = df[df["name"] == "Iris-setosa"]
# df_ver = df[df["name"] == "Iris-versicolor"]
# df_vir = df[df["name"] == "Iris-virginica"]

# for index, row in seta.iterrows():
# 	for x in range(len(feature)):

# 		if(index=="mean"):
# 			seta[feature[x]][index] = df_vir[feature[x]].mean()
# 		elif(index=="std"):
# 			seta[feature[x]][index] = df_vir[feature[x]].std()

# seta.to_csv('Iris-virginica.csv')
# #
actual_set_predict_set = 0
actual_set_predict_ver = 0
actual_set_predict_vir = 0
actual_ver_predict_set = 0
actual_ver_predict_ver = 0
actual_ver_predict_vir = 0
actual_vir_predict_set = 0
actual_vir_predict_ver = 0
actual_vir_predict_vir = 0





for l in range(1,4):
	if(l==1):
		### df1 = testing data
		df1 = df[:50]
		### df2 = training data
		df2 = df[50:]
	elif(l==2):
		### df1 = testing data
		df1 = df[50:100]
		### df2 = training data
		df2 = df[0:50]
		df2 = df2.append(df[100:150], ignore_index = True)
	elif(l==3):
		### df1 = testing data
		df1 = df[100:]
		### df2 = training data
		df2 = df[0:100]

	df_set = df2[df2["name"] == "Iris-setosa"]
	df_ver = df2[df2["name"] == "Iris-versicolor"]
	df_vir = df2[df2["name"] == "Iris-virginica"]

	record_mean = {"Iris-setosa":{},"Iris-versicolor":{},"Iris-virginica":{}}
	record_std = {"Iris-setosa":{},"Iris-versicolor":{},"Iris-virginica":{}}
	#Iris-setosa
	for x in range(4):
		record_mean["Iris-setosa"][feature[x]] = df_set[feature[x]].mean()
		record_std["Iris-setosa"][feature[x]] = df_set[feature[x]].std()

	#Iris-versicolor
	for x in range(4):
		record_mean["Iris-versicolor"][feature[x]] = df_ver[feature[x]].mean()
		record_std["Iris-versicolor"][feature[x]] = df_ver[feature[x]].std()

	#Iris-virginica
	for x in range(4):
		record_mean["Iris-virginica"][feature[x]] = df_vir[feature[x]].mean()
		record_std["Iris-virginica"][feature[x]] = df_vir[feature[x]].std()

	for index, row in df1.iterrows():
		probability_of_set = 0;
		probability_of_ver = 0;
		probability_of_vir = 0;
		for x in range(len(feature)):
			if(record_std["Iris-setosa"][feature[x]]!=0):
				probability_of_set += math.log(1/(record_std["Iris-setosa"][feature[x]]*math.sqrt(2*math.pi)) *math.exp((row[feature[x]]- record_mean["Iris-setosa"][feature[x]])**2 *(-1)/( 2*(record_std["Iris-setosa"][feature[x]])**2)  ))
			if(record_std["Iris-versicolor"][feature[x]]!=0):
				probability_of_ver += math.log(1/(record_std["Iris-versicolor"][feature[x]]*math.sqrt(2*math.pi)) *math.exp((row[feature[x]]- record_mean["Iris-versicolor"][feature[x]])**2 *(-1)/( 2*(record_std["Iris-versicolor"][feature[x]])**2)  ))
			if(record_std["Iris-virginica"][feature[x]]!=0):
				probability_of_vir += math.log(1/(record_std["Iris-virginica"][feature[x]]*math.sqrt(2*math.pi)) *math.exp((row[feature[x]]- record_mean["Iris-virginica"][feature[x]])**2 *(-1)/( 2*(record_std["Iris-virginica"][feature[x]])**2)  ))
		pset = df2[df2["name"]=="Iris-setosa"].shape[0] /df2.shape[0]
		pver = df2[df2["name"]=="Iris-versicolor"].shape[0] /df2.shape[0]
		pvir = df2[df2["name"]=="Iris-virginica"].shape[0] /df2.shape[0]
		probability_of_set += math.log(pset)
		probability_of_ver += math.log(pver)
		probability_of_vir += math.log(pvir)
		if( max(probability_of_set,probability_of_ver,probability_of_vir) ==probability_of_set and row["name"]=="Iris-setosa"):
			actual_set_predict_set+=1
		elif( max(probability_of_set,probability_of_ver,probability_of_vir) ==probability_of_ver and row["name"]=="Iris-setosa"):
			actual_set_predict_ver+=1
		elif( max(probability_of_set,probability_of_ver,probability_of_vir) ==probability_of_vir and row["name"]=="Iris-setosa"):
			actual_set_predict_vir+=1
		elif( max(probability_of_set,probability_of_ver,probability_of_vir) ==probability_of_set and row["name"]=="Iris-versicolor"):
			actual_ver_predict_set+=1
		elif( max(probability_of_set,probability_of_ver,probability_of_vir) ==probability_of_ver and row["name"]=="Iris-versicolor"):
			actual_ver_predict_ver+=1
		elif( max(probability_of_set,probability_of_ver,probability_of_vir) ==probability_of_vir and row["name"]=="Iris-versicolor"):
			actual_ver_predict_vir+=1
		elif( max(probability_of_set,probability_of_ver,probability_of_vir) ==probability_of_set and row["name"]=="Iris-virginica"):
			actual_vir_predict_set+=1
		elif( max(probability_of_set,probability_of_ver,probability_of_vir) ==probability_of_ver and row["name"]=="Iris-virginica"):
			actual_vir_predict_ver+=1
		elif( max(probability_of_set,probability_of_ver,probability_of_vir) ==probability_of_vir and row["name"]=="Iris-virginica"):
			actual_vir_predict_vir+=1	


print("K-fold cross-validation results: ")
print("-------------------------")
print("Confusion matrix:")
print("			   predict Iris-setosa          predict Iris-versicolor           predict Iris-virginica")
print("actual Iris-setosa:                ",int(actual_set_predict_set/3),"                    ",int(actual_set_predict_ver/3),"                   		 ",int(actual_set_predict_vir/3))
print("actual Iris-versicolor:             ",int(actual_ver_predict_set/3),"                    ",int(actual_ver_predict_ver/3),"                   		 ",int(actual_ver_predict_vir/3))
print("actual Iris-virginica:              ",int(actual_vir_predict_set/3),"                    ",int(actual_vir_predict_ver/3),"                   		 ",int(actual_vir_predict_vir/3))
print('\n')
print("Sensitivity(Recall): ")
print("			   predict Iris-setosa          predict Iris-versicolor           predict Iris-virginica")
print("				", actual_set_predict_set / (actual_set_predict_set+ actual_set_predict_ver+actual_set_predict_vir),"				",actual_ver_predict_ver / (actual_ver_predict_ver+ actual_ver_predict_set+actual_ver_predict_vir), "				",actual_vir_predict_vir / (actual_vir_predict_set+ actual_vir_predict_ver+actual_vir_predict_vir))
print('\n')
print("Precision: ")
print("			   predict Iris-setosa          predict Iris-versicolor           predict Iris-virginica")
print("				",actual_set_predict_set / (actual_set_predict_set+actual_ver_predict_set+actual_vir_predict_set),"				",actual_ver_predict_ver / (actual_ver_predict_ver+actual_set_predict_ver+actual_vir_predict_ver), "				",actual_vir_predict_vir / (actual_vir_predict_vir+actual_ver_predict_vir+actual_set_predict_vir))
print('\n')
print("Accuracy: ", (actual_set_predict_set + actual_ver_predict_ver + actual_vir_predict_vir) / (df1.shape[0]*3) )

print("-------------------------")
print('\n\n\n')


actual_set_predict_set = 0
actual_set_predict_ver = 0
actual_set_predict_vir = 0
actual_ver_predict_set = 0
actual_ver_predict_ver = 0
actual_ver_predict_vir = 0
actual_vir_predict_set = 0
actual_vir_predict_ver = 0
actual_vir_predict_vir = 0

df = df.sample(frac=1).reset_index(drop=True)
df1.iloc[0:0]
df2.iloc[0:0]
r0 = [random.randint(0, df.shape[0]) for _ in range(int(df.shape[0]/3))] 

for x in range(1,df.shape[0]):
	if x in r0:
		df1 = df1.append(df[x:x], ignore_index = True)
	else:
		df2 = df2.append(df[x:x], ignore_index = True)




df_set = df2[df2["name"] == "Iris-setosa"]
df_ver = df2[df2["name"] == "Iris-versicolor"]
df_vir = df2[df2["name"] == "Iris-virginica"]

record_mean = {"Iris-setosa":{},"Iris-versicolor":{},"Iris-virginica":{}}
record_std = {"Iris-setosa":{},"Iris-versicolor":{},"Iris-virginica":{}}
#Iris-setosa
for x in range(4):
	record_mean["Iris-setosa"][feature[x]] = df_set[feature[x]].mean()
	record_std["Iris-setosa"][feature[x]] = df_set[feature[x]].std()

#Iris-versicolor
for x in range(4):
	record_mean["Iris-versicolor"][feature[x]] = df_ver[feature[x]].mean()
	record_std["Iris-versicolor"][feature[x]] = df_ver[feature[x]].std()

#Iris-virginica
for x in range(4):
	record_mean["Iris-virginica"][feature[x]] = df_vir[feature[x]].mean()
	record_std["Iris-virginica"][feature[x]] = df_vir[feature[x]].std()

for index, row in df1.iterrows():
	probability_of_set = 0;
	probability_of_ver = 0;
	probability_of_vir = 0;
	for x in range(len(feature)):
		if(record_std["Iris-setosa"][feature[x]]!=0):
			probability_of_set += math.log(1/(record_std["Iris-setosa"][feature[x]]*math.sqrt(2*math.pi)) *math.exp((row[feature[x]]- record_mean["Iris-setosa"][feature[x]])**2 *(-1)/( 2*(record_std["Iris-setosa"][feature[x]])**2)  ))
		if(record_std["Iris-versicolor"][feature[x]]!=0):
			probability_of_ver += math.log(1/(record_std["Iris-versicolor"][feature[x]]*math.sqrt(2*math.pi)) *math.exp((row[feature[x]]- record_mean["Iris-versicolor"][feature[x]])**2 *(-1)/( 2*(record_std["Iris-versicolor"][feature[x]])**2)  ))
		if(record_std["Iris-virginica"][feature[x]]!=0):
			probability_of_vir += math.log(1/(record_std["Iris-virginica"][feature[x]]*math.sqrt(2*math.pi)) *math.exp((row[feature[x]]- record_mean["Iris-virginica"][feature[x]])**2 *(-1)/( 2*(record_std["Iris-virginica"][feature[x]])**2)  ))
	pset = df2[df2["name"]=="Iris-setosa"].shape[0] /df2.shape[0]
	pver = df2[df2["name"]=="Iris-versicolor"].shape[0] /df2.shape[0]
	pvir = df2[df2["name"]=="Iris-virginica"].shape[0] /df2.shape[0]
	probability_of_set += math.log(pset)
	probability_of_ver += math.log(pver)
	probability_of_vir += math.log(pvir)
	if( max(probability_of_set,probability_of_ver,probability_of_vir) ==probability_of_set and row["name"]=="Iris-setosa"):
		actual_set_predict_set+=1
	elif( max(probability_of_set,probability_of_ver,probability_of_vir) ==probability_of_ver and row["name"]=="Iris-setosa"):
		actual_set_predict_ver+=1
	elif( max(probability_of_set,probability_of_ver,probability_of_vir) ==probability_of_vir and row["name"]=="Iris-setosa"):
		actual_set_predict_vir+=1
	elif( max(probability_of_set,probability_of_ver,probability_of_vir) ==probability_of_set and row["name"]=="Iris-versicolor"):
		actual_ver_predict_set+=1
	elif( max(probability_of_set,probability_of_ver,probability_of_vir) ==probability_of_ver and row["name"]=="Iris-versicolor"):
		actual_ver_predict_ver+=1
	elif( max(probability_of_set,probability_of_ver,probability_of_vir) ==probability_of_vir and row["name"]=="Iris-versicolor"):
		actual_ver_predict_vir+=1
	elif( max(probability_of_set,probability_of_ver,probability_of_vir) ==probability_of_set and row["name"]=="Iris-virginica"):
		actual_vir_predict_set+=1
	elif( max(probability_of_set,probability_of_ver,probability_of_vir) ==probability_of_ver and row["name"]=="Iris-virginica"):
		actual_vir_predict_ver+=1
	elif( max(probability_of_set,probability_of_ver,probability_of_vir) ==probability_of_vir and row["name"]=="Iris-virginica"):
		actual_vir_predict_vir+=1	


print("Holdout validation results: ")
print("-------------------------")
print("Confusion matrix:")
print("			   predict Iris-setosa          predict Iris-versicolor           predict Iris-virginica")
print("actual Iris-setosa:                ",int(actual_set_predict_set),"                    ",int(actual_set_predict_ver),"                   		 ",int(actual_set_predict_vir))
print("actual Iris-versicolor:             ",int(actual_ver_predict_set),"                    ",int(actual_ver_predict_ver),"                   		 ",int(actual_ver_predict_vir))
print("actual Iris-virginica:              ",int(actual_vir_predict_set),"                    ",int(actual_vir_predict_ver),"                   		 ",int(actual_vir_predict_vir))
print('\n')
print("Sensitivity(Recall): ")
print("			   predict Iris-setosa          predict Iris-versicolor           predict Iris-virginica")
print("				", actual_set_predict_set / (actual_set_predict_set+ actual_set_predict_ver+actual_set_predict_vir),"				",actual_ver_predict_ver / (actual_ver_predict_ver+ actual_ver_predict_set+actual_ver_predict_vir), "				",actual_vir_predict_vir / (actual_vir_predict_set+ actual_vir_predict_ver+actual_vir_predict_vir))
print('\n')
print("Precision: ")
print("			   predict Iris-setosa          predict Iris-versicolor           predict Iris-virginica")
print("				",actual_set_predict_set / (actual_set_predict_set+actual_ver_predict_set+actual_vir_predict_set),"				",actual_ver_predict_ver / (actual_ver_predict_ver+actual_set_predict_ver+actual_vir_predict_ver), "				",actual_vir_predict_vir / (actual_vir_predict_vir+actual_ver_predict_vir+actual_set_predict_vir))
print('\n')
print("Accuracy: ", (actual_set_predict_set + actual_ver_predict_ver + actual_vir_predict_vir) / (df1.shape[0]) )

print("-------------------------")
print('\n\n\n')
