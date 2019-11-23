# hold out 
import numpy as np
import pandas as pd
import math
import random
import csv

df = pd.read_csv("X_train.csv")
df2 = pd.read_csv("y_train.csv")


max_depth = 20
depth = 0


cat = []
for num in df2["Category"]:
    cat.append(num)
df["category"] = cat


non_cts = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

for col in non_cts:
    df[col].replace([" ?"], [df[col].mode()], inplace = True)


attr = { "age":{">", "<"},
        "workclass":{" Private", " Self-emp-not-inc"," Self-emp-inc"," Federal-gov"," Local-gov"," State-gov"," Without-pay"," Never-worked"},
        "fnlwgt":{">", "<"},
        "education":{" Bachelors"," Some-college"," 11th"," HS-grad"," Prof-school"," Assoc-acdm"," Assoc-voc"," 9th"," 7th-8th"," 12th"," Masters"," 1st-4th"," 10th"," Doctorate"," 5th-6th"," Preschool"}, 
        "education-num":{">", "<"}, 
        "marital-status":{" Married-civ-spouse"," Divorced"," Never-married"," Separated"," Widowed"," Married-spouse-absent"," Married-AF-spouse"},
        "occupation":{" Tech-support"," Craft-repair"," Other-service"," Sales"," Exec-managerial"," Prof-specialty"," Handlers-cleaners"," Machine-op-inspct"," Adm-clerical"," Farming-fishing"," Transport-moving"," Priv-house-serv"," Protective-serv"," Armed-Forces"}, 
        "relationship":{" Wife"," Own-child"," Husband"," Not-in-family"," Other-relative"," Unmarried"}, 
        "race":{" White"," Asian-Pac-Islander"," Amer-Indian-Eskimo"," Other"," Black"}, 
        "sex":{" Female"," Male"},
        "capital-gain":{">", "<"},
        "capital-loss":{">", "<"},
        "hours-per-week":{">", "<"}, 
        "native-country":{" United-States"," Cambodia"," England"," Puerto-Rico"," Canada"," Germany"," Outlying-US(Guam-USVI-etc)"," India"," Japan"," Greece"," South"," China"," Cuba"," Iran"," Honduras"," Philippines"," Italy"," Poland"," Jamaica"," Vietnam"," Mexico"," Portugal"," Ireland"," France"," Dominican-Republic"," Laos"," Ecuador"," Taiwan"," Haiti"," Columbia"," Hungary"," Guatemala"," Nicaragua"," Scotland"," Thailand"," Yugoslavia"," El-Salvador"," Trinadad&Tobago"," Peru"," Hong"," Holand-Netherland"}
       }






df = df.sample(frac=1).reset_index(drop=True)

t = int (df.shape[0]/3)
df_test = df[t*2:]
df = df[0:2*t]

def gini_index_categorical(dataframe):
    count0 = dataframe[dataframe["category"]==0].shape[0]
    count1 = dataframe[dataframe["category"]==1].shape[0]
    if((count0+count1) !=0):
        return (1 - (count0/(count0+count1))**2 - (count1/(count0+count1))**2)
    return 0

def gini_index_cts(dataframe,str):
    t1 =0
    t2 =0
    df1 = dataframe[dataframe[str] > df[str].mean()]
    df2 = dataframe[dataframe[str] < df[str].mean()]
    count0 = df1[df1["category"]==0].shape[0]
    count1 = df1[df1["category"]==1].shape[0]
    if((count0+count1) !=0):
        t1 = (1 - (count0/(count0+count1))**2 - (count1/(count0+count1))**2)*df1.shape[0]/dataframe.shape[0]
    count0 = df2[df2["category"]==0].shape[0]
    count1 = df2[df2["category"]==1].shape[0]
    if((count0+count1) !=0):
        t2 = (1 - (count0/(count0+count1))**2 - (count1/(count0+count1))**2)*df2.shape[0]/dataframe.shape[0]
    return t1+t2




def get_attr_feature(df):
    max_gini = 0;
    select_attr = ""
    for str in attr:
        current_gini = gini_index_categorical(df)
        feature_gini = -999999
        current_num = df.shape[0]
        if(str == "age" or str == "fnlwgt" or str == "education-num" or str == "capital-gain" or str == "capital-loss" or str == "hours-per-week" ):
            feature_gini = gini_index_cts(df,str)
        else:
            for feature in attr[str]:
                df_tt = df[df[str]==feature];
                feature_gini +=  gini_index_categorical(df_tt)*(df_tt.shape[0]/current_num)
        if((current_gini - feature_gini)>max_gini):
            max_gini = (current_gini - feature_gini)
            select_attr = str
    return select_attr

def build_tree(df, attribute,depth):
    depth+=1
    if(almost_one(df)):
        return {"val": 1}
    elif(almost_zero(df)):
        return {"val": 0}
    elif(depth>max_depth):
        return {"val": check(df)}
    else:
        tree = {attribute:{}}
        for feature in attr[attribute]:
            tree[attribute][feature] = {feature: {}}
            if(feature != ">" and feature != "<"):
                dft = df[df[attribute]==feature];
            elif(feature ==">"):
                dft = df[df[attribute] > df[attribute].mean()];
            elif(feature =="<"):  
                dft = df[df[attribute] < df[attribute].mean()];
            if(dft.empty):
                return {"val": check(df)}
            next_attr = get_attr_feature(dft)
            subtree = build_tree(dft, next_attr,depth)
            tree[attribute][feature] = subtree
        return tree

def check(df):
    t = 0
    f = 0
    for s in df["category"]==1:
        if(s):
            t +=1
        elif(not s):
            f+=1
    if(t>=f):
        return 1
    else:
        return 0

def almost_one(df):
    t = 0
    f = 0
    for s in df["category"]==1:
        if(s):
            t +=1
        elif(not s):
            f+=1
    if(t/(t+f) > 0.999):
        return 1
    else:
        return 0
    
def almost_zero(df):
    t = 0
    f = 0
    for s in df["category"]==1:
        if(s):
            t +=1
        elif(not s):
            f+=1
    if(f/(t+f) > 0.999):
        return 1
    else:
        return 0
    
def predict(df,tree):
    test = {}
    for index, row in df.iterrows():
        tree2 = tree
        key = next(iter(tree2))
        while key != 'val':
            if(key == ' ?'):
                test[row["Id"]] =  random.randint(1,10) %2
                continue
            tree2 = tree2[key][row[key]]
            key = next(iter(tree2))
        test[row["Id"]] = tree2[key]
    return test


def predict_hold(df,tree):
    a0p0 = 0 
    a0p1 = 0
    a1p0 = 0
    a1p1 = 0
    for index, row in df.iterrows():
        tree2 = tree
        key = next(iter(tree2))
        while key != 'val':
            if(key == ' ?'):
                if(row["category"]==0):
                    a0p0 +=1
                continue
            tree2 = tree2[key][row[key]]
            key = next(iter(tree2))
        if(tree2[key]==0 and row["category"]==0):
            a0p0 +=1
        elif(tree2[key]==0 and row["category"]==1):
            a1p0 +=1
        elif(tree2[key]==1 and row["category"]==0):
            a0p1 +=1
        elif(tree2[key]==1 and row["category"]==1):
            a1p1 +=1
    return a0p0, a0p1, a1p0, a1p1

tree = build_tree(df, get_attr_feature(df), 0)
a0_p0, a0_p1, a1_p0, a1_p1 = predict_hold(df_test,tree)


print("Hold-out cross-validation results  ")
print("-------------------------")
print("Confusion matrix:")
print("             predict 1   predict 0")
print("actual 1:    ",a1_p1,"        ",a1_p0)
print("actual 0:    ",a0_p1,"        ",a0_p0)
print('\n')
print("	    predict 1                predict 0")
print("Recall:  ",a1_p1 / (a1_p1+a1_p0), "  ",a0_p0/(a0_p0+a0_p1) )
print('\n')
print("	        predict 1                predict 0")
print("Precision:   ",a1_p1 / (a1_p1+a0_p1), "   ",a0_p0/(a0_p0+a1_p0))
print('\n')
print("Accuracy: " ,(a1_p1 + a0_p0) / (a1_p1+a0_p0+a1_p0+a0_p1) )
print("-------------------------")
print('\n\n\n')

