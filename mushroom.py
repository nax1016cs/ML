import numpy as np
import pandas as pd
import math
import random

df = pd.read_csv("mushroom.data", header = None, names = ["edible","cap-shape","cap-surface","cap-color","bruises?","odor","gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring","veil-type","veil-color","ring-number","ring-type","spore-print-color","population","habitat"])

df = df[df["stalk-root"] != '?']


split = []
index_ct = 0
feature = ["edible","cap-shape","cap-surface","cap-color","bruises?","odor","gill-attachment","gill-spacing","gill-size","gill-color","stalk-shape","stalk-root","stalk-surface-above-ring","stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring","veil-type","veil-color","ring-number","ring-type","spore-print-color","population","habitat"]

### record the index of spliting df into 3 parts
for index, row in df.iterrows():
	index_ct +=1
	if(index_ct == int(df.shape[0]/3)):
		split.append(index_ct)
		index_ct = 0;

for i in range(1,3):
	split[i]+=split[i-1]


type_of_feature = ["a","b","c","d","e","f","g","h","k","l","m","n","o","p","r","s","t","u","v","w","x","y"] 

condition_df_e = pd.DataFrame(index=feature, columns = type_of_feature)
condition_df_p = pd.DataFrame(index=feature, columns = type_of_feature)
condition_df_total = pd.DataFrame(index=feature, columns = type_of_feature)
df = df.sample(frac=1).reset_index(drop=True)


### record the number of correct cases without laplace
ae_pe = 0 #actual e and predict e
ae_pp = 0 #actual e and predict p
ap_pe = 0 #actual p and predict e
ap_pp = 0 #actual p and predict p

### record the number of correct cases with laplace
ae_pe_lp = 0 #actual e and predict e
ae_pp_lp = 0 #actual e and predict p
ap_pe_lp = 0 #actual p and predict e
ap_pp_lp = 0 #actual p and predict p

		


### 3-fold 
for i in range(1,4):
	if(i==1):
		### df1 = testing data
		df1 = df[:split[0]]
		### df2 = training data
		df2 = df[split[0]:]

	elif(i==2):
		### df1 = testing data
		df1 = df[split[0]:split[1]]
		### df2 = training data
		df2 = df[split[1]:]
		df2 = df2.append(df[:split[0]], ignore_index = True)

	elif(i==3):
		### df1 = testing data
		df1 = df[split[1]+1:]
		### df2 = training data
		df2 = df[:split[1]]




	dfe = df2[df2["edible"] == 'e']
	dfp = df2[df2["edible"] == 'p']


	k=5
	number_of_feature = [2,6,4,10,2,9,4,3,2,12,2,6,4,4,9,9,2,4,3,8,9,6,7]

	for index, row in condition_df_p.iterrows():
		for x in range(len(type_of_feature)):
			# condition_df_e[type_of_feature[x]][index] = ((dfe[dfe[index] == type_of_feature[x]].shape[0])+k )/ (fe_size+k*number_of_feature[x])
			# condition_df_p[type_of_feature[x]][index] = ((dfp[dfp[index] == type_of_feature[x]].shape[0])+k )/ (fp_size+k*number_of_feature[x])
			condition_df_e[type_of_feature[x]][index] = ((dfe[dfe[index] == type_of_feature[x]].shape[0]))
			condition_df_p[type_of_feature[x]][index] = ((dfp[dfp[index] == type_of_feature[x]].shape[0]))
	# find the target of every feature
	for index, row in df1.iterrows():
		
		probability_of_e = 0
		probability_of_p = 0

		probability_of_e_lp = 0
		probability_of_p_lp = 0
		for x in range(len(feature)):
			if(x>0):
				target = row[feature[x]]
				
				## caculate the probability of P(x|'e')
				## no laplace
				if(condition_df_e[target][feature[x]] !=0):
					probability_of_e += math.log((condition_df_e[target][feature[x]])/(dfe.shape[0]))

				if(condition_df_p[target][feature[x]] !=0):
					probability_of_p += math.log((condition_df_p[target][feature[x]])/(dfp.shape[0]))

				# laplace 
				probability_of_e_lp += math.log((condition_df_e[target][feature[x]]+k)/(dfe.shape[0]+k*number_of_feature[x]))
				probability_of_p_lp += math.log((condition_df_p[target][feature[x]]+k)/(dfp.shape[0]+k*number_of_feature[x]))
				

		### prbability of p(e) and p(p)			
		pe = dfe.shape[0] / df2.shape[0]
		pp = dfp.shape[0] / df2.shape[0]
		## no laplace
		probability_of_e += math.log(pe)
		probability_of_p += math.log(pp)
		#laplace
		probability_of_e_lp += math.log(pe)
		probability_of_p_lp += math.log(pp)

		#caculate without laplace
		if(probability_of_e >= probability_of_p and row['edible']=='e'):
			ae_pe +=1
		elif(probability_of_e > probability_of_p and row['edible']=='p'):
			ap_pe+=1
		elif(probability_of_e < probability_of_p and row['edible']=='e'):
			ae_pp+=1
		elif(probability_of_e < probability_of_p and row['edible']=='p'):
			ap_pp+=1

		#caculate with laplace
		if(probability_of_e_lp >= probability_of_p_lp and row['edible']=='e'):
			ae_pe_lp +=1
		elif(probability_of_e_lp > probability_of_p_lp and row['edible']=='p'):
			ap_pe_lp+=1
		elif(probability_of_e_lp < probability_of_p_lp and row['edible']=='e'):
			ae_pp_lp+=1
		elif(probability_of_e_lp < probability_of_p_lp and row['edible']=='p'):
			ap_pp_lp+=1



print("K-fold cross-validation results without laplace: ")
print("-------------------------")
print("Confusion matrix:")
print("			   predict edible             predict poisonable")
print("actual edible:                ",int(ae_pe/3),"                    	",int(ae_pp/3))
print("actual poisonable:            ",int(ap_pe/3),"                    		",int(ap_pp/3))
print('\n')
print("			   edible             			poisonable")
print("Recall:                ",ae_pe / (ae_pe+ae_pp), "        ",ap_pp/(ap_pp+ap_pe) )
print('\n')
print("			   edible             			poisonable")
print("Precision:             ",ae_pe / (ae_pe+ap_pe), "        ",ap_pp/(ap_pp+ae_pp))
print('\n')
print("Accuracy: " ,(ae_pe + ap_pp) / (ae_pe+ap_pp+ae_pp+ap_pe) )
print("-------------------------")
print('\n\n\n')


print("K-fold cross-validation results with laplace: ")
print("-------------------------")
print("Confusion matrix:")
print("			   predict edible             predict poisonable")
print("actual edible:                ",int(ae_pe_lp/3),"                    	",int(ae_pp_lp/3))
print("actual poisonable:            ",int(ap_pe_lp/3),"                    		",int(ap_pp_lp/3))
print('\n')
print("			   edible             		poisonable")
print("Recall:                ",ae_pe_lp / (ae_pe_lp+ae_pp_lp), "        ",ap_pp_lp/(ap_pp_lp+ap_pe_lp) )
print('\n')
print("			   edible             		poisonable")
print("Precision:             ",ae_pe_lp / (ae_pe_lp+ap_pe_lp), "        ",ap_pp_lp/(ap_pp_lp+ae_pp_lp))
print('\n')
print("Accuracy: ", (ae_pe_lp + ap_pp_lp) / (ae_pe_lp+ap_pp_lp+ae_pp_lp+ap_pe_lp) )
print("-------------------------")
print('\n\n\n')




### record the number of correct cases without laplace
ae_pe = 0 #actual e and predict e
ae_pp = 0 #actual e and predict p
ap_pe = 0 #actual p and predict e
ap_pp = 0 #actual p and predict p

### record the number of correct cases with laplace
ae_pe_lp = 0 #actual e and predict e
ae_pp_lp = 0 #actual e and predict p
ap_pe_lp = 0 #actual p and predict e
ap_pp_lp = 0 #actual p and predict p

df = df.sample(frac=1).reset_index(drop=True)

df1.iloc[0:0]
df2.iloc[0:0]

r0 = [random.randint(0, df.shape[0]) for _ in range(int (df.shape[0]/3))] 

for x in range(1,df.shape[0]):
	if x in r0:
		df1 = df1.append(df[x:x], ignore_index = True)
	else:
		df2 = df2.append(df[x:x], ignore_index = True)


dfe = df2[df2["edible"] == 'e']
dfp = df2[df2["edible"] == 'p']

for index, row in condition_df_p.iterrows():
	for x in range(len(type_of_feature)):
		# condition_df_e[type_of_feature[x]][index] = ((dfe[dfe[index] == type_of_feature[x]].shape[0])+k )/ (fe_size+k*number_of_feature[x])
		# condition_df_p[type_of_feature[x]][index] = ((dfp[dfp[index] == type_of_feature[x]].shape[0])+k )/ (fp_size+k*number_of_feature[x])
		condition_df_e[type_of_feature[x]][index] = ((dfe[dfe[index] == type_of_feature[x]].shape[0]) )
		condition_df_p[type_of_feature[x]][index] = ((dfp[dfp[index] == type_of_feature[x]].shape[0]) )

for index, row in df1.iterrows():
		
	probability_of_e = 0
	probability_of_p = 0

	probability_of_e_lp = 0
	probability_of_p_lp = 0
	for x in range(len(feature)):
		if(x>0):
			target = row[feature[x]]
			
			## caculate the probability of P(x|'e')
			## no laplace
			if(condition_df_e[target][feature[x]] !=0):
				probability_of_e += math.log((condition_df_e[target][feature[x]])/(dfe.shape[0]))

			if(condition_df_p[target][feature[x]] !=0):
				probability_of_p += math.log((condition_df_p[target][feature[x]])/(dfp.shape[0]))

			# laplace 
			probability_of_e_lp += math.log((condition_df_e[target][feature[x]]+k)/(dfe.shape[0]+k*number_of_feature[x]))
			probability_of_p_lp += math.log((condition_df_p[target][feature[x]]+k)/(dfp.shape[0]+k*number_of_feature[x]))
			

	### prbability of p(e) and p(p)			
	pe = dfe.shape[0] / df2.shape[0]
	pp = dfp.shape[0] / df2.shape[0]
	## no laplace
	probability_of_e += math.log(pe)
	probability_of_p += math.log(pp)
	#laplace
	probability_of_e_lp += math.log(pe)
	probability_of_p_lp += math.log(pp)

	#caculate without laplace
	if(probability_of_e >= probability_of_p and row['edible']=='e'):
		ae_pe +=1
	elif(probability_of_e > probability_of_p and row['edible']=='p'):
		ap_pe+=1
	elif(probability_of_e < probability_of_p and row['edible']=='e'):
		ae_pp+=1
	elif(probability_of_e < probability_of_p and row['edible']=='p'):
		ap_pp+=1

	#caculate with laplace
	if(probability_of_e_lp >= probability_of_p_lp and row['edible']=='e'):
		ae_pe_lp +=1
	elif(probability_of_e_lp > probability_of_p_lp and row['edible']=='p'):
		ap_pe_lp+=1
	elif(probability_of_e_lp < probability_of_p_lp and row['edible']=='e'):
		ae_pp_lp+=1
	elif(probability_of_e_lp < probability_of_p_lp and row['edible']=='p'):
		ap_pp_lp+=1

print("\n\n\n")
print("Holdout-validation results without laplace: ")
print("-------------------------")
print("Confusion matrix:")
print("			   predict edible             predict poisonable")
print("actual edible:                ",int(ae_pe),"                    	",int(ae_pp))
print("actual poisonable:            ",int(ap_pe),"                    		",int(ap_pp))
print('\n')
print("			   edible             			poisonable")
print("Recall:                ",ae_pe / (ae_pe+ae_pp), "        ",ap_pp/(ap_pp+ap_pe) )
print('\n')
print("			   edible             			poisonable")
print("Precision:             ",ae_pe / (ae_pe+ap_pe), "        ",ap_pp/(ap_pp+ae_pp))
print('\n')
print("Accuracy: " ,(ae_pe + ap_pp) / (ae_pe+ap_pp+ae_pp+ap_pe) )
print("-------------------------")
print('\n\n\n')



print("Holdout-validation results with laplace: ")
print("-------------------------")
print("Confusion matrix:")
print("			   predict edible             predict poisonable")
print("actual edible:                ",int(ae_pe_lp),"                    	",int(ae_pp_lp))
print("actual poisonable:            ",int(ap_pe_lp),"                    		",int(ap_pp_lp))
print('\n')
print("			   edible             			poisonable")
print("Recall:                ",ae_pe_lp / (ae_pe_lp+ae_pp_lp), "        ",ap_pp_lp/(ap_pp_lp+ap_pe_lp) )
print('\n')
print("			   edible             			poisonable")
print("Precision:             ",ae_pe_lp / (ae_pe_lp+ap_pe_lp), "        ",ap_pp_lp/(ap_pp_lp+ae_pp_lp))
print('\n')
print("Accuracy: ", (ae_pe_lp + ap_pp_lp) / (ae_pe_lp+ap_pp_lp+ae_pp_lp+ap_pe_lp) )
print("-------------------------")
print('\n\n\n')

