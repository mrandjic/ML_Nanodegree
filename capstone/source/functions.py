import numpy as np
import pandas as pd
import datetime as dt
from time import time
import visuals as vs
import matplotlib.pyplot as plt
from IPython.display import display # Allows the use of display() for DataFrames
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, fbeta_score, accuracy_score

def naive_predictor(data):
    tp = float(len(data[(data['NAIVE_PRED']=="TARGET") & (data['FLAG']=="TARGET")]))
    tn = float(len(data[(data['NAIVE_PRED']=="NO_TARGET") & (data['FLAG']=="NO_TARGET")]))
    fp = float(len(data[(data['NAIVE_PRED']=="TARGET") & (data['FLAG']=="NO_TARGET")]))
    fn = float(len(data[(data['NAIVE_PRED']=="NO_TARGET") & (data['FLAG']=="TARGET")]))

    #Calculate precision
    precision = tp/(tp+fp)
    
    #Calculate recall
    recall    = tp/(tp+fn)

    #Calculate accuracy
    accuracy = (tp+tn)/(tp+tn+fp+fn)

    #Calculate F-score using the formula above for beta = 2
    beta = 2
    fscore = (1+np.square(beta))*(precision*recall)/(np.square(beta)*precision+recall)

    # Print the results 
    print("TP={}".format(tp))
    print("FP={}".format(fp))
    print("FN={}".format(fn))
    print("TN={}".format(tn))
    print("Precision={}".format(precision))
    print("Recall={}".format(recall))
    print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)
	
	
# Perform One-Hot Encoding to categorical features 
def encode_features(data_set): 
    
    TIME_ID = data_set['TIME_ID'] 
    FLAG    = data_set['FLAG'] 
    
    data_set = data_set.drop(['TIME_ID','FLAG'], axis=1)
    data_set['ACT_MONTH'] = data_set.ACT_MONTH.astype(str)
    data_set = pd.get_dummies(data_set)
    
    data_set = pd.concat([data_set, TIME_ID], axis=1)#.reset_index(drop=True)
    data_set = pd.concat([data_set, FLAG], axis=1)#.reset_index(drop=True)
    
    encoded = list(data_set.columns)
    print ("{} total features after one-hot encoding.".format(len(encoded)))
    
    return(data_set)
	
# Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.metrics import confusion_matrix

def train_predict(learner, sample_frac, train_set, test_set): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    num_of_samples = 30000
    
    train_set_no_target = pd.DataFrame(train_set[(train_set['FLAG'] == 0)])
    train_set_target    = train_set[(train_set['FLAG'] == 1)]

    train_set_no_target = train_set_no_target.groupby(['TIME_ID'])#, 'SERVICE_SEG'
    train_set_no_target = train_set_no_target.apply(lambda x: x.sample(frac=sample_frac/100.0))

    train_set = pd.concat([train_set_no_target, train_set_target])

    #Shuffle train set
    train_set = train_set.sample(frac=1)
    
    train_set_features = train_set.drop(['FLAG', 'TIME_ID', 'SUBSCRIPTION_ID'], axis = 1)
    train_set_labels   = train_set['FLAG']
    
    test_set_features  = test_set.drop(['FLAG', 'TIME_ID', 'SUBSCRIPTION_ID'], axis = 1)
    test_set_labels    = test_set['FLAG']
    
    # Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner.fit(train_set_features, train_set_labels)
    end = time() # Get end time
    
    # Calculate the training time
    results['train_time'] = end - start
        
    # Get the predictions on the test set,
    #       then get predictions on the first 30000 training samples
    start = time() # Get start time
    predictions_test = learner.predict(test_set_features)
    predictions_train = learner.predict(train_set_features[:num_of_samples])
    end = time() # Get end time
    
    tn, fp, fn, tp = confusion_matrix(test_set_labels,predictions_test).ravel()#

    tn=float(tn)
    fp=float(fp)
    fn=float(fn)
    tp=float(tp)
    
    precision = tp/(tp+fp)
    recall    = tp/(tp+fn)
	
	#Calculate accuracy
    accuracy = (tp+tn)/(tp+tn+fp+fn)
	
	#Calculate F-score using the formula above for beta = 2
    beta = 2
    fscore = (1+np.square(beta))*(precision*recall)/(np.square(beta)*precision+recall)

    # Calculate the total prediction time
    results['pred_time'] = end-start
            
    # Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score(train_set_labels[:num_of_samples],predictions_train)
    
    # Compute accuracy on test set
    results['acc_test'] = accuracy_score(test_set_labels,predictions_test)
    
    # Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(train_set_labels[:num_of_samples],predictions_train,beta=2)
    
    # Compute F-score on the test set
    results['f_test'] = fbeta_score(test_set_labels,predictions_test,beta=2)
       
    # Success
    print ("{} trained on {} % of dataset with {} samples.".format(learner.__class__.__name__, sample_frac, train_set_labels.shape[0]))
    print ("tn={}, fp={}, fn={}, tp={}, precision={}, recall={}, accuracy={}, fscore={}.".format( tn, fp, fn, tp, precision, recall, accuracy, fscore ))
    
    
    # Return the results
    return results
	
def generate_test_sets(data, start_test_dt, end_test_dt): 
    
	#Generate separate test sets for each month from selected date range
    temp_dt = start_test_dt
    test_sets_features = {}
    test_sets_labels   = {}
    test_sets_ID       = {}

    index = 1

    while temp_dt <= end_test_dt:

        temp_test_set = pd.DataFrame(data[(data['TIME_ID'] == temp_dt) ])

        #put preprocessed test set to collection
        test_sets_features[index] = temp_test_set.drop(['FLAG', 'TIME_ID', 'SUBSCRIPTION_ID'], axis = 1)
        test_sets_labels[index]   = temp_test_set['FLAG']
        test_sets_ID[index]       = temp_test_set['SUBSCRIPTION_ID']

        temp_year  = temp_dt.year
        temp_month = temp_dt.month

        if temp_month == 12: 
            temp_year = temp_year + 1

        temp_month = (temp_dt.month+1)%12
        if temp_month == 0:
            temp_month=12

        temp_dt = dt.datetime(temp_year, temp_month, temp_dt.day).date()
        index = index + 1
		
    #Extract three sets: input features, target labels, and subscription IDs
    dataframe_collection = {} 
    dataframe_collection[1] = test_sets_features
    dataframe_collection[2] = test_sets_labels
    dataframe_collection[3] = test_sets_ID
    
    return(dataframe_collection)

	
def generate_train_set(data_set, start_train_dt, end_train_dt, sample_frac): 
    
    data_set = data_set[(data_set['TIME_ID'] >= start_train_dt) & (data_set['TIME_ID'] <= end_train_dt)]
    
    #Shuffle data set
    data_set = data_set.sample(frac=1)
    
    data_set_no_target = pd.DataFrame(data_set[(data_set['FLAG'] == 0)])
    data_set_target    = data_set[(data_set['FLAG'] == 1)]
    
    print("Initial size of no_target class: {}".format(len(data_set_no_target)))
    
	#Downscale NO_TARGET group
    data_set_no_target = data_set_no_target.groupby(['TIME_ID'])
    data_set_no_target = data_set_no_target.apply(lambda x: x.sample(frac=sample_frac))
    
    print("Downscaled size of no_target class: {}".format(len(data_set_no_target)))    
	
    print("Target class size: {}".format(len(data_set_target)))

    data_set = pd.concat([data_set_no_target, data_set_target])
    
    features = data_set.drop(['FLAG', 'TIME_ID', 'SUBSCRIPTION_ID'], axis = 1)
    labels   = data_set['FLAG']
    
	# Return features and labels of downscaled train set
    train_set = {}
    train_set[1] = features
    train_set[2] = labels
    
    return(train_set)
	
def test_predict(test_month_index, clf, test_sets_features, test_sets_labels, test_sets_ID): 
    #generate probabilities of classification
	pred_prob  = clf.predict_proba(test_sets_features[test_month_index])
    #generate predicted class
	pred_class = clf.predict(test_sets_features[test_month_index])
	
	#generate confusion matrix
    tn, fp, fn, tp = confusion_matrix(test_sets_labels[test_month_index], pred_class).ravel()

    tn=float(tn)
    fp=float(fp)
    fn=float(fn)
    tp=float(tp)
    
    #Calculate precision
    precision = tp/(tp+fp)
    
    #Calculate recall
    recall    = tp/(tp+fn)

    #Calculate accuracy
    accuracy = (tp+tn)/(tp+tn+fp+fn)

    #Calculate F-score using the formula above for beta = 2
    beta = 2
    fscore = (1+np.square(beta))*(precision*recall)/(np.square(beta)*precision+recall)

    # Print the results 
    print("TP={}".format(tp))
    print("FP={}".format(fp))
    print("FN={}".format(fn))
    print("TN={}".format(tn))
    print("Precision={}".format(precision))
    print("Recall={}".format(recall))
    print "ML Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)

    return(pd.DataFrame({'C0_SCORE':pred_prob.T[0],'C1_SCORE':pred_prob.T[1], 'PRED_CLASS':pred_class, 'ACT_CLASS':test_sets_labels[test_month_index], 'SUBSCRIPTION_ID':test_sets_ID[test_month_index]}))
	

def campaign_simulation(target_data, activations_data, act_class, pred_class, score_treshold ):
    
	#Define from which group (FP or TP), target list for the t+1 month will be created, in month t.
	#ACT_CLASS is the actual class in month t
	#PRED_CLASS is the predicted class in month t
	#Knowing those two vectors, we can select TP or FP users in order to observe whether users from one of these groups will activate the service in t+1 month
	#Minimum score of prediction is 0.5. 
	#Score threshold can be increased, in order to track take rate and roaming rate.
	
    target_list    = target_data[ (target_data.ACT_CLASS==act_class) & (target_data.PRED_CLASS==pred_class) & (target_data.C1_SCORE>score_treshold)]
    
	#Extract IDs of all users from the month t+1
    roamers        = pd.DataFrame(activations_data['SUBSCRIPTION_ID'])
    
	#Match users from target list to list of roamers, in order to find number of actual roaming users from previous month t
    actual_roamers = pd.merge(target_list, roamers, on=['SUBSCRIPTION_ID'], how='inner')
    
	#Match users that activated add-on service
    activations    = activations_data[activations_data.ACT_CLASS==1]
	
	#Select IDs only
    activations    = pd.DataFrame(activations['SUBSCRIPTION_ID'])
	
	#Find actual takers from our target list
    activations    = pd.merge(target_list, activations, on=['SUBSCRIPTION_ID'], how='inner')
    
	#Plot score histogram of takers that were correctly classified by our algorithm
    plt
	
	
    fig = plt.figure(figsize = (15,5));
    ax = fig.add_subplot(1, 2, 2)
    ax.hist(activations['C1_SCORE'])
    ax.set_title('SCORING DISTRIBUTION', fontsize = 14)
    ax.set_xlabel("Score")
    ax.set_ylabel("Number customers")
    
	#Print campaign statistics
    print("Total number of targeted users: \t\t\t{}".format( len(target_list)))
    print("Total number of roaming users in targeted month: \t{}".format( len(roamers)))
    print("Total number of correctly targeted roaming users: \t{}".format( len(actual_roamers)))
    print("Correctly targeted roaming users rate: \t\t\t{}%".format( float(len(actual_roamers))/float(len(target_list))*100))
    print("Total number of activations: \t\t\t\t{}".format( len(activations)))
    print("Take rate: \t\t\t\t\t\t{}%".format( float(len(activations))/float(len(actual_roamers))*100))
	
