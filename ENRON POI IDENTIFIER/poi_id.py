#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier

import pandas as pd
import matplotlib.pyplot as plt
import pylab

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#All features
features_list = ['poi','salary','deferral_payments','total_payments','exercised_stock_options','bonus','restricted_stock','restricted_stock_deferred','total_stock_value','expenses','loan_advances','other','director_fees','deferred_income','long_term_incentive','percentage_from_poi','percentage_to_poi','percentage_shared_receipt_from_poi']

#All features except aggreagted features toatl_payments and total_stock_value
#features_list = ['poi','salary','deferral_payments','exercised_stock_options','bonus','restricted_stock','restricted_stock_deferred','expenses','loan_advances','other','director_fees','deferred_income','long_term_incentive','percentage_from_poi','percentage_to_poi','percentage_shared_receipt_from_poi']

#Only aggregated financial features and email related features
#features_list = ['poi','total_payments','total_stock_value','percentage_from_poi','percentage_to_poi','percentage_shared_receipt_from_poi']



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#print individual names in the dataset
def display_people(data_dict):
    for person in data_dict.keys():
        print person


#validation derived attribute calculation
def validate_derived_attr(data_dict):
    calculated_total_payments=0
    calculated_total_stock_value=0
    incorrect_total_payments=[]
    incorrect_total_stock_value=[]
    for person,attributes in data_dict.iteritems():
        calculated_total_payments=attributes['salary']+attributes['bonus']+attributes['director_fees']+attributes['deferral_payments']+attributes['deferred_income']+attributes['loan_advances']+attributes['long_term_incentive']+attributes['expenses']+attributes['other']
        calculated_total_stock_value=attributes['restricted_stock']+attributes['exercised_stock_options']+attributes['restricted_stock_deferred']
        if attributes['total_payments']!=calculated_total_payments:
            incorrect_total_payments.append(person)
        if attributes['total_stock_value']!=calculated_total_stock_value:
            incorrect_total_stock_value.append(person)
    print "individuals with incorrect total_payments:"
    print incorrect_total_payments
    print "individuals with incorrect total_stock_value:"
    print incorrect_total_stock_value
    
    
#function to report basic statistics
def basic_stat(data_dict):
    count_total=0
    count_poi=0
    for person,attributes in data_dict.iteritems():
        count_total=count_total+1
        if attributes['poi']==True:
            count_poi=count_poi+1
    print "Total_Count:",count_total
    print "POI_Count:",count_poi

#function to plot the correlation_matrix
def plot_corr(df,size=8):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns,rotation=90);
    plt.yticks(range(len(corr.columns)), corr.columns);
    pylab.show()

#exploring the data
def data_exploration(data_dict,feature1,feature2):
    poi_count=0
    non_poi_count=0
    for person,attributes in data_dict.iteritems():
        if person!="TOTAL" and attributes[feature1]!='NaN' and attributes[feature2]!='NaN':
            color="b"
            if attributes['poi']==True:
                color="r"
                poi_count=poi_count+1
                if poi_count==1:
                    plt.scatter(attributes[feature1],attributes[feature2],color=color,label="poi")
            else:
                non_poi_count+=1
                if non_poi_count==1:
                    plt.scatter(attributes[feature1],attributes[feature2],color=color,label="non-poi")
            plt.scatter(attributes[feature1],attributes[feature2],color=color)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()
    print "poi count:",  poi_count
    print "non poi count",non_poi_count

#Function to replace NaN's with zeros
def replace_NaN(data_dict):
    for person,attributes in data_dict.iteritems():
        for key,value in attributes.iteritems():
            if value=='NaN':
                attributes[key]=0
    return data_dict


### Task 2: Remove outliers
del(data_dict["TOTAL"])
#display_people(data_dict)
del(data_dict["THE TRAVEL AGENCY IN THE PARK"])

'''
# Report basic statistics
#basic_stat(data_dict)

#replace NaN
#data_dict=replace_NaN(data_dict)

# Validate derived attributes
#validate_derived_attr(data_dict)
'''
#correct data for BELFER ROBERT and BHATNAGAR SANJAY using enron61702insiderpay.
data_dict['BELFER ROBERT']['director_fees']=102500
data_dict['BELFER ROBERT']['deferral_payments']=0
data_dict['BELFER ROBERT']['deferred_income']=-102500
data_dict['BELFER ROBERT']['expenses']=3285
data_dict['BELFER ROBERT']['total_payments']=3285
data_dict['BELFER ROBERT']['restricted_stock']=44093
data_dict['BELFER ROBERT']['exercised_stock_options']=0
data_dict['BELFER ROBERT']['restricted_stock_deferred']=-44093
data_dict['BELFER ROBERT']['total_stock_value']=0

data_dict['BHATNAGAR SANJAY']['director_fees']=0
data_dict['BHATNAGAR SANJAY']['expenses']=137864
data_dict['BHATNAGAR SANJAY']['other']=0
data_dict['BHATNAGAR SANJAY']['total_payments']=137864
data_dict['BHATNAGAR SANJAY']['restricted_stock']=2604490
data_dict['BHATNAGAR SANJAY']['exercised_stock_options']=15456290
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred']=-2604490
data_dict['BHATNAGAR SANJAY']['total_stock_value']=15456290


### Task 3: Create new feature(s)
for person,attributes in data_dict.iteritems():
    if attributes["to_messages"]=='NaN':
        attributes["percentage_from_poi"]='NaN'
        attributes["percentage_to_poi"]='NaN'
        attributes["percentage_shared_receipt_from_poi"]='NaN'
    else:
        attributes["percentage_from_poi"]=attributes["from_poi_to_this_person"]/float(attributes["to_messages"])
        attributes["percentage_to_poi"]=attributes["from_this_person_to_poi"]/float(attributes["from_messages"])
        attributes["percentage_shared_receipt_from_poi"]=attributes["shared_receipt_with_poi"]/float(attributes["to_messages"])


### Store to my_dataset for easy export below.
my_dataset = replace_NaN(data_dict)


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


#Getting the feature scores
'''
anova_feature_filter=SelectKBest(f_classif,k='all')
anova_feature_filter.fit(features,labels)
print anova_feature_filter.scores_
'''
#Correlation Matrix
'''
df_features=pd.DataFrame(data=features,columns=features_list[1:])
plot_corr(df_features)
'''

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

#Naive_Bayes

from sklearn.naive_bayes import GaussianNB
NB_clf=GaussianNB()

#SVM
'''
from sklearn.svm import SVC
svc_clf = SVC(kernel='rbf',C=1,gamma=.001)
'''
#Decision Trees
'''
from sklearn.tree import DecisionTreeClassifier
dtree_clf=DecisionTreeClassifier(random_state=42)
'''

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# trying various values for min_samples_split and checking the performance
##NB Tuning

params_NB = dict(anova__k=[1,2,3,4,5,6,7,8,9,10])
anova_feature_filter=SelectKBest(f_classif,k=10)
anova_NB=Pipeline([('anova',anova_feature_filter),('NB',NB_clf)])


##SVC tuning

'''
#params_SVC=dict(anova__k=[1,2,3,4,5,6,7,8,9,10])
anova_feature_filter=SelectKBest(f_classif,k='all')
scaler=StandardScaler()
anova_svc=Pipeline([('anova',anova_feature_filter),('scaler',scaler),('svc',svc_clf)])
'''


##DecisionTree Tuning
'''
params_dtree = dict(anova__k=[1,2,3,4,5,6,7,8,9,10],dtree__min_samples_split=[1,2,3,4,5,6,7,8,9,10])
anova_feature_filter=SelectKBest(f_classif,k=10)
anova_dtree=Pipeline([('anova',anova_feature_filter),('dtree',dtree_clf)])
'''


from sklearn.cross_validation import StratifiedShuffleSplit
folds=1000
cv = StratifiedShuffleSplit(labels, folds, random_state = 42)

scoring_metric = 'recall'
GridSearch=GridSearchCV(anova_NB,param_grid=params_NB,scoring=scoring_metric,cv=cv)
#GridSearch=GridSearchCV(anova_dtree,param_grid=params_dtree,scoring=scoring_metric,cv=cv)

GridSearch.fit(features,labels)
#print "features_importances:"
#print GridSearch.feature_importances_
clf=GridSearch.best_estimator_
print clf
#print "scores:"
#print anova_feature_filter.scores_

#clf=anova_dtree

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)



