'''
Libraries to install : 
    - matplotlib
    - numpy
    - pandas
    - sklearn 
Try "pip install library_name" to install
'''

import warnings
from os import system, name
import matplotlib.pyplot as plt
from random import random, seed
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics, linear_model, preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

def clear(): #Clearing consol 
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
  
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear') 

def warn(*args, **kwargs): #Removing sklearn warnings
    pass

def initialisation(url="https://pulse-events.fr/esme/pfe/cs-training.csv", index_col=0):
    df = pd.read_csv(url)
    le = df.shape[0]
    df = df.dropna()
    return df

def correlation_matrix(methods="spearman", df=[]):
    # methods = {'pearson', 'kendall', 'spearman'}
    matrix_corr = df.corr(method=methods)["SeriousDlqin2yrs"]
    return matrix_corr.sort_values(ascending=False)

def rescaling(df):
    scaler = preprocessing.MinMaxScaler()
    scaled_df = scaler.fit_transform(df)
    return scaled_df

def splitting(df, feature_columns, target, random_value=1):
    split_train = 0.50
    sample_train = 1-split_train
    size = int(df.shape[0])
    size_train = int(size*0.30)
    size_test = int(size*0.70)
    X_train = df[feature_columns][:size_train].sample(
        n=int((size*split_train)*sample_train), random_state=random_value)
    X_train = rescaling(X_train)
    Y_train = df[target][:size_train].sample(
        n=int((size*split_train)*sample_train), random_state=random_value)
    X_test = df[feature_columns][:size_test]
    X_test = rescaling(X_test)
    Y_test = df[target][:size_test]
    return [X_train,Y_train,X_test,Y_test]

def neural_network(df, feature_columns, target, random_value=1, step=[20,3]):
    split = splitting(df, feature_columns, target, random_value=1)
    X_train = split[0]
    Y_train = split[1]
    X_test = split[2]
    Y_test = split[3]

    mlp = MLPClassifier(hidden_layer_sizes=(step),
                        activation='relu', solver='adam', max_iter=500)
    mlp.fit(X_train, Y_train)
    predict_train = mlp.predict(X_train)
    predict_test = mlp.predict(X_test)

    # print(confusion_matrix(Y_test, predict_test))
    classifi = classification_report(Y_test, predict_test, output_dict=True)
    #print(classifi)
    if(step==[20,3]):
        print(classification_report(Y_test, predict_test))
    # print(classification_report(Y_test, predict_test, output_dict=True))
    return [classifi['1']['precision'], classifi['1']['recall']]

def sam(df, feature_columns, target, random_value=1, max_iters=[5, 5, 5],graph=True):
    nn = []
    for i in range(0, max_iters[0]+1):
        for j in range(0, max_iters[1]+1):
            for k in range(0,max_iters[1]+1):
                st = [i,j,k]
                if(j == 0):
                    st = [i]
                if(i!=0 and j != 0 and k==0):
                    st = [i,j]
                try:
                    results = neural_network(df, feature_columns, target, random_value=1,step=st)
                    print("Iteration Typo : (",st,") : ", results)
                    nn.append([st,results[0],results[1]])
                except:
                    pass
                if(j == 0 or k==0):
                    break
                #          [typo],precision,recall
    nn = pd.DataFrame(nn,columns=['typo','precision','recall'])
    nn = nn.dropna()
    nn.to_csv("export.csv", encoding='utf-8', index=False)
    max_precision =  nn.loc[nn['precision'].idxmax()]
    max_recall = nn.loc[nn['recall'].idxmax()]
    print("")
    print("MAX PRECISION : ")
    print("TYPO : ",max_precision['typo']," Precision : ",max_precision['precision']," - Recall : ",max_precision['recall'])
    print("MAX RECALL : ")
    print("TYPO : ",max_recall['typo']," Precision : ",max_recall['precision']," - Recall : ",max_recall['recall'])
    
    if(graph and max_iters[1]==0 and max_iters[1]==0):
        plt.figure(figsize=(9, 3))
        plt.plot([i for i in range(max_iters[0])],nn['precision'],color='blue')
        plt.plot([i for i in range(max_iters[0])],nn['recall'],color='orange')
        plt.suptitle('Precision (blue) and recall (orange) scores ')
        plt.show()
    return 1

def logistic_regression(df, feature_columns, target, random_value=1):
    split = splitting(df, feature_columns, target, random_value=1)
    X_train = split[0]
    Y_train = split[1]
    X_test = split[2]
    Y_test = split[3]

    regr=linear_model.LogisticRegression(solver='liblinear')

    regr.fit(X_train, Y_train)

    prediction=regr.predict(X_test)
    rec_score=classification_report(Y_test, prediction)
    # print(classification_report(Y_test, prediction))
    return [regr.coef_, rec_score]

def sampling_logistic_regression(df, target, feature_columns, graph=False, iterations_arg=2):
    coef=logistic_regression(
        df, feature_columns, target, int(10*random()))[0]
    coef=coef[0]
    iteration=1
    max_iteration=iterations_arg
    coef_sampling=[]
    rand_value=[int(100*random()) for i in range(max_iteration)]
    prec_rec=[]
    for i in range(max_iteration):
        iteration += 1
        fun2=logistic_regression(df, feature_columns, target, rand_value[i])
        coef=((coef*iteration)+fun2[0]/(iteration+1))
        # coef_sampling.append(np.around(coef[0], decimals=20))
        coef_sampling.append(coef[0])

        print("Iteration : ", i)
    print("Confusion Matrix", (prec_rec))
    print(coef)
    print(fun2[1])
    if(graph):
        sampled=[[] for i in range(len(feature_columns))]
        for y in range(len(feature_columns)):
            for item in coef_sampling:
                sampled[y].append(item[y])
        fig, axs=plt.subplots(2, 5,
                                sharey=True, tight_layout=True)
        i=0
        for f in range(2):
            for a in range(5):
                try:
                    axs[f][a].hist(sampled[i], bins=200)
                    axs[f][a].set_title(feature_columns[i][0:15], wrap=True)
                    if(i == len(feature_columns)):
                        i=0
                    i=i+1
                except:
                    i=i
        # plt.tight_layout()
        plt.show()

    return 1

def decision_tree(df, feature_columns, target,max_dep = 2):
    split = splitting(df, feature_columns, target, random_value=1)
    X_train = split[0]
    Y_train = split[1]
    X_test = split[2]
    Y_test = split[3]
    tree = DecisionTreeClassifier(max_depth = max_dep)
    tree.fit(X_train,Y_train)
    prediction = tree.predict(X_test)
    print(classification_report(Y_test, prediction))
    return 1


if __name__ == "__main__":
    warnings.warn = warn #Removing warnings from sklearn
    clear()
    print("Initialisation...")
    df=initialisation()
    target="SeriousDlqin2yrs"
    # Columns used as variables
    feature_columns=['NumberOfTimes90DaysLate',
                    'NumberOfTime60-89DaysPastDueNotWorse',
                    'NumberOfTime30-59DaysPastDueNotWorse',
                    'RevolvingUtilizationOfUnsecuredLines',
                    'NumberOfDependents',
                    'DebtRatio',
                    'NumberRealEstateLoansOrLines',
                    'NumberOfOpenCreditLinesAndLoans',
                    'MonthlyIncome',
                    'age']
    loop = True
    while(loop):
        loop = False
        clear()
        print("## Choix du modèle ##")
        print("1. Logistic Regression")
        print("2. Neural Network")
        print("3. Decision Tree")
        print("4. Correlation Matrix")
        print("Type enter to exit")
        choice = int(input())
        if(choice != 4):
            print("# Methode")
            print("1. Execution unique")
            print("2. Echantillonnage")
            method = int(input())
            met = "Echantillonnage"
            if( method == 1):
                met = "Execution unique"
        clear()
        print(" ")
        if(choice == 1):
            print("# Logistic Regression - ",met)
            if(method == 2):
                iterations = int(input("Combien d'itérations"))
            else:
                iterations = 1 
            sampling_logistic_regression(df, target, feature_columns, True, iterations_arg = iterations)
        elif(choice == 2):
            print("# Neural Network - ",met)
            if(method == 2):
                iterations = int(input("2 Layers - Combien de noeuds max par layer ?"))
                print("Typologies testées : [1,1] ->  [",iterations,",",iterations,"]")
                sam(df, feature_columns, target, random_value=1,max_iters=[iterations,iterations])
            else:
                print("Typologie testée : [20,3]")
                iterations = [20,3]
                neural_network(df, feature_columns, target, random_value=1)
            print("")
        elif(choice == 3):
            print("# Decision Tree - ",met)
            if(method == 2):
                print("Profondeur max : ")
                iterations = int(input())
            else:
                iterations = 2
            decision_tree(df, feature_columns, target,max_dep = iterations)
        elif(choice == 4):
             print("# Correlation Matrix - Pearson Method")
             print(correlation_matrix("pearson", df))
        print("")
        loop = input("Type Enter to exit, 1 to start again ")
        if(loop != "1"):
            loop == True
        

