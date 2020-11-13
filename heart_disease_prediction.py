import sys
import csv
import math
import numpy as np
from operator import itemgetter
import time

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC 
from sklearn.externals import joblib
from sklearn.feature_selection import RFE, VarianceThreshold, SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, chi2
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import KBinsDiscretizer, scale

#Handle annoying warnings
import warnings, sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

#############################################################################
#
# Global parameters
#
#################################################

model_technique = "MLPC"				            #The technique to explore for the experiment. The options are "MLR", "RF", "MLPC", "SVM". 
target_idx=13                                       #Index of Target variable
cross_val=1                                         #Control Switch for CV                                                                                      
norm_target=0                                       #Normalize target switch
norm_features=1                                     #Normalize target switch
binning=1                                           #Control Switch for Bin Target
bin_cnt=2                                           #If bin target, this sets number of classes
feat_select=1                                       #Control Switch for Feature Selection
fs_type=2                                           #Feature Selection type (1=Stepwise Backwards Removal, 2=Wrapper Select, 3=Univariate Selection)
lv_filter=0                                         #Control switch for low variance filter on features
feat_start=0                                        #Start column of features
k_cnt=5 
                                            #Number of 'Top k' best ranked features to select, only applies for fs_types 1 and 3
#Set global model parameters
rand_st=1 


#Printing all the settings
print("\n")

#Normalization of data 
if norm_target == 1:
    print("The target is normalized")

if norm_features == 1:
    print("The independant features are normalized")


#Low- Variance 
if lv_filter==1:
    print("Low Variance Filter applied")
elif lv_filter==0:
    print("No Low Variance Filter applied")

#Feature Selection 
if feat_select ==1:
    #Feature Selection - Type 
    if fs_type==1:
        print("Feature Selection applied - Stepwise Backwards Removal")
    elif fs_type==2:
        print("Feature Selection applied - Wrapper Select")
    elif fs_type==3: 
        if binning == 0:
            print("Feature Selection Applied - Univariate Selection - Mutual Info")
        elif binning == 1:
            print("Feature Selection Applied - Univariate Selection - Chi2")
		
    elif fs_type==4:
	    print("Feature Selection applied - Univariate Selection") 	#Set Random State variable for randomizing splits on runs
	
else:
    print ("No Feature Selection")

	
#Machine Learning Technique used 	
if model_technique == "RF":
    print("Random Forest Technique applied")
elif model_technique == "MLR":
    print("Multivariate Logistic Regression applied")
elif model_technique == "MLPC":
    print("Neural Network Technique: Multi Layer Perceptron applied")
else:
    print ("Super Vector Machines applied")

#Train-Test Split  
if cross_val == 0:
    print("Train/Test Split : 35/65 applied")
#Cross - Validation
elif cross_val == 1:
    print("K - Fold Cross Validation applied")


#############################################################################
#
# Load Data
#
#####################

file1= csv.reader(open('heart.csv'), delimiter=',', quotechar='"')

#Read Header Line
header=next(file1)          

#Read data
data=[]
target=[]
for row in file1:
    #Load Target
    if row[target_idx]=='':                         #If target is blank, skip row                       
        continue
    else:
        target.append(float(row[target_idx])) 		#If pre-binned class, change float to int
    #Load row into temp array, cast columns  
    temp=[]
                 
    for j in range(feat_start,len(header)-1):
        if row[j]=='':
            temp.append(float())
        else:
            temp.append(float(row[j]))

    #Load temp into Data array
    data.append(temp)
  
#Test Print
print("\n", "Features: " + str(header))
print("\n", "Length of target: " +str(len(target)), "Length of data: " +str(len(data)))
print('\n')

data_np=np.asarray(data)
target_np=np.asarray(target)

#############################################################################
#
# Preprocess data
#
##########################################

#Normalization of data 
if norm_target==1:
    #Target normalization for continuous values
    target_np=scale(target_np)

if norm_features==1:
    #Feature normalization for continuous values
    data_np=scale(data_np)


#############################################################################
#
# Feature Selection
#
##########################################

#Low Variance Filter
if lv_filter==1:
    print('--LOW VARIANCE FILTER ON--', '\n')
    
    #LV Threshold
    sel = VarianceThreshold(threshold=0.5)                                          #Removes any feature with less than 20% variance
    fit_mod=sel.fit(data_np)
    fitted=sel.transform(data_np)
    sel_idx=fit_mod.get_support()

    #Get lists of selected and non-selected features (names and indexes)
    temp=[]
    temp_idx=[]
    temp_del=[]
    for i in range(len(data_np[0])):
        if sel_idx[i]==1:                                                           #Selected Features get added to temp header
            temp.append(header[i+feat_start])
            temp_idx.append(i)
        else:                                                                       #Indexes of non-selected features get added to delete array
            temp_del.append(i)

    print('Selected:', temp)
    print('Features (total, selected):', len(data_np[0]), len(temp))
    print('\n')

    #Filter selected columns from original dataset
    header = header[0:feat_start]
    for field in temp:
        header.append(field)
    data_np = np.delete(data_np, temp_del, axis=1)                                 #Deletes non-selected features by index
		
#Feature Selection
if feat_select==1:
    '''Three steps:
       1) Run Feature Selection
       2) Get lists of selected and non-selected features
       3) Filter columns from original dataset
       '''
    
    print('--FEATURE SELECTION ON--', '\n')
    
    
    ##1) Run Feature Selection #######
    if fs_type==1:
        #Stepwise Recursive Backwards Feature removal
        if binning==1:
            clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=3, criterion='entropy', random_state=rand_st)
            sel = RFE(clf, n_features_to_select=k_cnt, step=.1)
            print('Stepwise Recursive Backwards - Random Forest: ')
        if binning==0:
            rgr = RandomForestRegressor(n_estimators=500, max_depth=None, min_samples_split=3, criterion='mse', random_state=rand_st)
            sel = RFE(rgr, n_features_to_select=k_cnt, step=.1)
            print('Stepwise Recursive Backwards - Random Forest: ')
            
        fit_mod=sel.fit(data_np, target_np)
        print(sel.ranking_)
        sel_idx=fit_mod.get_support()
		
    #Wrapper Select via model
    if fs_type==2:
	    #For Logistic Regression
        if model_technique == 'MLR':
            clf = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', max_depth = None, min_samples_split = 3, random_state = rand_st)
	    #For Random Forests
        if model_technique == 'RF':
            clf = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', max_depth = None, min_samples_split = 3, random_state = rand_st)		 
		#For Gradient Boosting
        elif model_technique == 'MLPC':
            clf = GradientBoostingClassifier(n_estimators=100, loss='deviance', learning_rate=0.1, max_depth=3, min_samples_split=3, random_state=rand_st)	
        #For SVM
        elif model_technique == 'SVM':
            clf = SVC(C=1.0, kernel='linear', degree=3, gamma='scale', probability=True,random_state=rand_st)
        		
        sel = SelectFromModel(clf, prefit=False, threshold='mean', max_features=None)                   
        print ('Wrapper Select: ')

        fit_mod=sel.fit(data_np, target_np)    
        sel_idx=fit_mod.get_support()
		
    if fs_type==3:
        if binning==1:                                                              ######Only work if the Target is binned###########
            #Univariate Feature Selection - Chi-squared
            sel=SelectKBest(chi2, k=k_cnt)
            fit_mod=sel.fit(data_np, target_np)                                         #will throw error if any negative values in features, so turn off feature normalization, or switch to mutual_info_classif
            print ('Univariate Feature Selection - Chi2: ')
            sel_idx=fit_mod.get_support()

        if binning==0:                                                              ######Only work if the Target is continuous###########
            #Univariate Feature Selection - Mutual Info Regression
            sel=SelectKBest(mutual_info_regression, k=k_cnt)
            fit_mod=sel.fit(data_np, target_np)
            print ('Univariate Feature Selection - Mutual Info: ')
            sel_idx=fit_mod.get_support()

        #Print ranked variables out sorted
        temp=[]
        scores=fit_mod.scores_
        for i in range(feat_start, len(header)-1):            
            temp.append([header[i], float(scores[i-feat_start])])

        print('Ranked Features')
        temp_sort=sorted(temp, key=itemgetter(1), reverse=True)
        for i in range(len(temp_sort)):
            print(i, temp_sort[i][0], ':', temp_sort[i][1])
        print('\n')
		
    if fs_type==4:
        clf = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', max_depth = None, min_samples_split = 3, random_state = rand_st)
        clf.fit(data_np, target_np)
        sel_idx = []
        mean = np.mean(clf.feature_importances_)
        for x in clf.feature_importances_:
            if x >= mean:
             sel_idx.append(1)
            else:
             sel_idx.append(0)
        
        

    ##2) Get lists of selected and non-selected features (names and indexes) #######
    temp=[]
    temp_idx=[]
    temp_del=[]
    for i in range(len(data_np[0])):
        if sel_idx[i]==1:                                                           #Selected Features get added to temp header
            temp.append(header[i+feat_start])
            temp_idx.append(i)
        else:                                                                       #Indexes of non-selected features get added to delete array
            temp_del.append(i)
    print('Selected:', temp)
    print('Features (total/selected):', len(data_np[0]), len(temp))
    print('\n')
            
               
    ##3) Filter selected columns from original dataset #########
    header = header[0:feat_start]
    for field in temp:
        header.append(field)
    data_np = np.delete(data_np, temp_del, axis=1)                                 #Deletes non-selected features by index

#############################################################################
#
# Train SciKit Models
#
##########################################

print('\n','--ML Model Output--', '\n')	
	
#Test/Train split
data_train, data_test, target_train, target_test = train_test_split(data_np, target_np, test_size=0.35)

####Classifiers####

###MULTIVARIATE LOGISTIC REGRESSION###
if model_technique == "MLR":
    #Train and Test Split Classifiers
    if cross_val == 0:
	    #Scikit Linear Logistic Regression
        clf = LogisticRegression(C = 1.0, penalty = 'l2', solver = 'lbfgs', max_iter=1000, random_state = rand_st)        
        clf.fit(data_train, target_train)
	
        scores_ACC = clf.score(data_test, target_test)                                                                                                                          
        print('Random Forest Acc:', scores_ACC)
        scores_AUC = metrics.roc_auc_score(target_test, clf.predict_proba(data_test)[:,1])                                                                                      
        print('Random Forest AUC:', scores_AUC)
		
	#Cross-Validation Classifier 
    if cross_val == 1:
	    #Setup Crossval classifier scorers
        scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'}
		
	    #SciKit Random Forest - Cross Val
        start_ts=time.time()
        clf = LogisticRegression(C = 1.0, penalty = 'l2', solver = 'liblinear', max_iter=1000, random_state = rand_st)   
        scores = cross_validate(clf, data_np, target_np, scoring = scorers, cv = 5)                                                                                                 
		
        scores_Acc = scores['test_Accuracy']                                                                                                                                    
        print("Random Forest Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                                                                    
        scores_AUC= scores['test_roc_auc']                                                                     #Only works with binary classes, not multiclass                  
        print("Random Forest AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                           
        print("CV Runtime:", time.time()-start_ts)
		
		
###RANDOM FORESTS###
if model_technique == "RF":
    #Train and Test Split Classifiers
    if cross_val==0:
	    #Scikit Random Forest 
        clf = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', max_depth = None, min_samples_split = 3, random_state = rand_st)
        clf.fit(data_train, target_train)
		
        scores_ACC = clf.score(data_test, target_test)                                                                                                                          
        print('Random Forest Acc:', scores_ACC)
        scores_AUC = metrics.roc_auc_score(target_test, clf.predict_proba(data_test)[:,1])                                                                                      
        print('Random Forest AUC:', scores_AUC)
		
	#Cross-Val Classifier
    if cross_val==1:
	    #Setup Crossval classifier scorers
        scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'}
		
	    #SciKit Random Forest - Cross Val
        start_ts=time.time()
        clf = RandomForestClassifier(n_estimators = 50, max_depth = None, min_samples_split = 3, criterion = 'entropy', random_state = rand_st)   
        scores = cross_validate(clf, data_np, target_np, scoring = scorers, cv = 5)                                                                                                 
		
        scores_Acc = scores['test_Accuracy']                                                                                                                                    
        print("Random Forest Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                                                                    
        scores_AUC= scores['test_roc_auc']                                                                     #Only works with binary classes, not multiclass                  
        print("Random Forest AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                           
        print("CV Runtime:", time.time()-start_ts)
		                                                                    

###NEURAL NETWORKS###
if model_technique == "MLPC":
    #Train/Test split
    if binning == 1 and cross_val == 0:
        clf = MLPClassifier(activation='logistic', solver='adam', alpha=0.0001, max_iter=1000, hidden_layer_sizes=(10,), random_state=rand_st)
        clf.fit(data_train, target_train)
		
        scores_ACC = clf.score(data_test, target_test)                                                                                                                          
        print('Random Forest Acc:', scores_ACC)
        scores_AUC = metrics.roc_auc_score(target_test, clf.predict_proba(data_test)[:,1])                                                                                      
        print('Random Forest AUC:', scores_AUC)
	
    if binning == 1 and cross_val == 1:
        #Setup Crossval classifier scorers
        scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'}
        
        #SciKit Neural Network - Cross Val
        start_ts=time.time()
        clf= MLPClassifier(activation='logistic', solver='adam', alpha=0.0001, max_iter=1000, hidden_layer_sizes=(100,50,20,), random_state=rand_st)
        scores= cross_validate(clf, data_np, target_np, scoring = scorers, cv = 5)

        scores_Acc = scores['test_Accuracy']                                                                                                                                    
        print("Neural Network Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                                                                    
        scores_AUC= scores['test_roc_auc']                                                                     #Only works with binary classes, not multiclass                  
        print("Neural Network AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                           
        print("CV Runtime:", time.time()-start_ts)		


##SUPER VECTOR MACHINES##
if model_technique == 'SVM':
    if binning == 1 and cross_val == 0:
        clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', probability=True,random_state=rand_st)
        clf.fit(data_train, target_train)
		
        scores_ACC = clf.score(data_test, target_test)                                                                                                                          
        print('SVM Acc:', scores_ACC)
        scores_AUC = metrics.roc_auc_score(target_test, clf.predict_proba(data_test)[:,1])                                                                                      
        print('SVM Forest AUC:', scores_AUC)

    if binning == 1 and cross_val == 1:
	    #Setup Crossval classifier scorers
        scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'}                                                                                                                
    
        #SciKit SVM - Cross Val
        start_ts=time.time()
        clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', probability=True,random_state=rand_st)
        scores = cross_validate(clf, data_np, target_np, scoring = scorers, cv = 10)

        scores_Acc = scores['test_Accuracy']                                                                                                                                    
        print("SVM Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                                                                    
        scores_AUC= scores['test_roc_auc']                                                                     #Only works with binary classes, not multiclass                  
        print("SVM AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                           
        print("CV Runtime:", time.time()-start_ts)
   
		




