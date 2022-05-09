import header as h
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

"""This function groups data by PID, so the sequences appear by PID instead of by order, in case it was interrupted by other PID
It returns a dict with PID as key and syscall seq as item"""
def group_syscalls_by_pid (data):
    seq_per_pid = {}
    for p in data['PID'].unique():
      filt = data['PID'] == p
      seq = data.loc[filt]['Syscall'].values.astype(str)
      seq_per_pid[p] = ' '.join(seq)

    seq_df = h.pd.DataFrame.from_dict(seq_per_pid, orient = 'index', columns=['Syscall Sequence'] )

    return seq_df


""" Parse an entire Syscall seq per PID into smaller sequences of size n """
def parse_seq(seq_per_pid):
    sequences = h.pd.DataFrame()
    for _, row in seq_per_pid.iterrows():
      token = h.word_tokenize(row['Syscall Sequence'])  # Tokenize the string of sequence

      # Parse the sequence into length of n
      sequences = h.pd.concat([sequences, h.pd.DataFrame(h.ngrams(token, 25, pad_right=True, right_pad_symbol=-1))] )
      # print('PID %d - seq len: %d'% (p, len(sequences)))
    return sequences


def preprocess_data(normal, intrusion):
  # Group system calls by PID
  normal_seq    = group_syscalls_by_pid(normal)
  intrusion_seq = group_syscalls_by_pid(intrusion)

  # Parse sys call sequence into n-grams n = 25
  normal    =  parse_seq(normal_seq)
  intrusion =  parse_seq(intrusion_seq)

  # Add labels to data
  normal['Label'] = 0;
  intrusion['Label'] = 1

  # Data partition 70/30
  df = h.pd.DataFrame()
  df = h.pd.concat([normal, intrusion], ignore_index=True).astype(int)  # append normal data and intrusion data into a df
  print('Df sz:', df.shape)

  # Partition data. x_train and x_test still have the labels with it, so no need to return y_train and y_test
  x_train, x_test, y_train, y_test = h.train_test_split(df, df['Label'], test_size=0.30, shuffle=True,
                                                        stratify=df['Label'])
  # Reset index of training and testing sets
  x_train.reset_index(drop=True, inplace=True);
  y_train.reset_index(drop=True, inplace=True)
  x_test.reset_index(drop=True, inplace=True);
  y_test.reset_index(drop=True, inplace=True)

  return x_train, x_test

def clean_test(train, test):
    # Convert train and test to list, then convert each sequence to a tupple. convert the tupples into a set
    train_list = train.values.tolist()
    test_list = test.values.tolist()
    train_set = set(tuple(i) for i in train_list)
    test_set = set(tuple(i) for i in test_list)
    print('List sz vs. Set sz of training sequences: %d vs. %d' % (len(train_list), len(train_set)))
    print('List sz vs. Set sz of testing sequences: %d vs. %d' % (len(test_list), len(test_set)))

    train_dupplication = (len(train_list) - len(train_set)) / len(train_list) * 100
    test_duplication = (len(test_list) - len(test_set)) / len(test_list) * 100

    print('Duplication Rate in training set: %.3f%%' % train_dupplication)
    print('Duplication Rate in test set: %.3f%%' % test_duplication)

    intersect = train_set.intersection(test_set)
    overlap_rate = len(intersect) * 100 / (len(train_set.union(test_set)))
    print('Overlap rate is %.3f%%' % overlap_rate)
    intersection_df = h.pd.DataFrame.from_dict(intersect).rename(columns={25: 'Label'})
    independent_test = h.pd.merge(intersection_df, test, how='outer', indicator=True).query('_merge=="right_only"').drop(
      columns='_merge')

    return independent_test


# Separate normal and intrusion in Test Clean so that I can call func clean_data on them
def separate_two_classes (data):
    ## Filter normal data from Test and drop label column
    filt = data.loc[:, 'Label'] == 0
    normal_class = data.loc[filt].copy()
    normal_class.drop(columns = 'Label', inplace = True)

    ## Filter Intrusion data from Test and drop label column
    intrusion_class = data.loc[~filt].copy()
    intrusion_class.drop(columns = 'Label', inplace = True)
    return normal_class, intrusion_class


# Convert normal df to set, and intrusion df to set, then remove overlapped sequences between both classes from
# intrusion class based on anomaly detection definition.
# Calculate duplication rate and overlap rate
def clean_data_between_two_classes(data):
    # separate two classes from the df
    normal, intrusion = separate_two_classes(data)

    # convert each class to set
    normal_list = normal.values.tolist()
    intrusion_list = intrusion.values.tolist()
    normal_set = set(tuple(i) for i in normal_list)
    intrusion_set = set(tuple(i) for i in intrusion_list)
    print('List sz vs. Set sz of normal sequences: %d vs. %d' % (len(normal_list), len(normal_set)))
    print('List sz vs. Set sz of intrusion sequences: %d vs. %d' % (len(intrusion_list), len(intrusion_set)))

    normal_duplication = (len(normal_list) - len(normal_set)) / len(normal_list) * 100
    intrusion_duplication = (len(intrusion_list) - len(intrusion_set)) / len(intrusion_list) * 100

    print('Duplication Rate in Normal Class: %.3f%%' % normal_duplication)
    print('Duplication Rate in Intrusion Class: %.3f%%' % intrusion_duplication)

    c_intrusion = intrusion_set - normal_set
    overlap_rate = len(normal_set.intersection(intrusion_set)) / len(normal_set.union(intrusion_set)) * 100
    print('Overlap rate: %.3f%%' % overlap_rate)

    if len(c_intrusion) == 0:
      print('Completely overlap! Using intrusion set as clean intrusion')
      intrusion = h.pd.DataFrame(intrusion_set)
    elif len(c_intrusion) > 0:
      intrusion = h.pd.DataFrame(c_intrusion)

    normal = h.pd.DataFrame(normal_set)
    print('After cleaning: \nNormal sz:', len(normal), ' Intrusion sz:', len(c_intrusion))

    # Add label back to data
    normal['Label'] = 0
    intrusion['Label'] = 1
    clean_data =  h.pd.concat([normal, intrusion], ignore_index=True) #test_clean_normal.append(test_clean_intrusion, ignore_index=True)
    clean_data = clean_data.sample(frac=1).reset_index(drop=True)  # Shuffle data

    return clean_data



"""Save each performance measure to a separate file. This version is different from HIDS_project"""
def write_stats(msg, metric, DATA_DIR, DATA):
    # Create test folder to store the duplication test files if it doesn't exist
    if not h.os.path.exists(DATA_DIR+'significance_test/'):
        h.os.makedirs(DATA_DIR+'significance_test/')

    print('writing to file: ', DATA_DIR+'significance_test/{}-{}-Stats.csv'.format(DATA,metric))
    outfile = open(DATA_DIR+'significance_test/{}-{}-Stats.csv'.format(DATA, metric), "a")  # Live-Named-F1-Stats.csv
    outfile.write(msg)  # model_name,train_status, macrof1
    outfile.close()


def calc_false_positive(cmatrix):
    specificity = cmatrix[0, 0] / (cmatrix[0, 0] + cmatrix[0, 1])
    return 1 - specificity


"""This function prints performance metrics and ROC curve given the model name, true labels and predicted labels"""
def print_performance( model_name, true_labels, pred_labels, model_clean_status, DATA_DIR, DATA):
    # rows are actual, columns are predicted
    cmatrix = h.confusion_matrix(true_labels, pred_labels)
    fpr = calc_false_positive(cmatrix)

    outfile = open(DATA_DIR+'significance_test/{}-Output.txt'.format(DATA), "a")  # Live-Named-Output.txt
    outfile.write('\nConfusion Matrix: \n' + str(cmatrix))
    outfile.write('\nTesting Accuracy: %.2f' % h.metrics.accuracy_score(true_labels, pred_labels))
    outfile.write('\nPrecision:%.2f' % h.metrics.precision_score(true_labels, pred_labels))
    outfile.write('\nRecall: %.2f' % h.metrics.recall_score(true_labels, pred_labels))
    outfile.write('\nFalse Positive Rate: %.2f' % fpr)
    outfile.write('\nClassification report:\n' + str(h.classification_report(true_labels, pred_labels)))
    outfile.write('AUC: %.2f \n\n' % h.roc_auc_score(true_labels, pred_labels))
    outfile.close()

    # model_name,train_status, macrof1
    msg = '%s,%s,%.2f\n' % (model_name, model_clean_status, h.metrics.f1_score(true_labels, pred_labels, average='macro'))
    write_stats(msg, 'F1', DATA_DIR, DATA)

    msg = '%s,%s,%.2f\n' % (model_name, model_clean_status,  h.roc_auc_score(true_labels, pred_labels))
    write_stats(msg, 'AUC', DATA_DIR, DATA)

    msg = '%s,%s,%.2f\n' % (model_name, model_clean_status, fpr)
    write_stats(msg, 'FPR', DATA_DIR, DATA)

    msg = '%s,%s,%.2f\n' % (model_name, model_clean_status, 1 - h.metrics.recall_score(true_labels, pred_labels))
    write_stats(msg, 'FNR', DATA_DIR, DATA)

    print('Confusion Matrix: \n', cmatrix)
    print('\nTesting Accuracy: %.2f' % h.metrics.accuracy_score(true_labels, pred_labels))
    print('Precision:%.2f' % h.metrics.precision_score(true_labels, pred_labels))
    print('Recall: %.2f' % h.metrics.recall_score(true_labels, pred_labels))
    print('False Positive Rate: %.2f' % fpr)
    print('\nClassification report:', h.classification_report(true_labels, pred_labels), sep='\n')
    print('AUC: %.2f' % h.roc_auc_score(true_labels, pred_labels))


    false_positive_rate, recall, thresholds = h.roc_curve(true_labels, pred_labels)
    roc_auc = h.auc(false_positive_rate, recall)
    plt.figure()
   # if CLEAN:
   #     clean_status = 'Clean '
   # else:
   #     clean_status = 'Overlapped and Duplicated '
    plt.title(model_name + ' ROC Curve on ' + model_clean_status + DATA + ' with Seq Len of 25')
    plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.1])
    plt.ylabel('Recall')
    plt.xlabel('False Positive Rate (1-Specificity)')
    # plt.savefig(model_name+'-ROC.jpg')
    #plt.show()


""" This func takes in Test sets to evaluate model. Make it convenient when testing with clean and unclean data"""
def test_model(data, label, model, model_name, test_clean_status, model_clean_status, DATA_DIR, DATA):
    """ Use this func to evaluate ML models"""
    y_predicted = model.predict(data)

    outfile = open(DATA_DIR +
                   'significance_test/{}-Output.txt'.format(DATA), "a")  # Live-Named-Output.txt
    outfile.write('\n--------------------{} {} on {} data --------------------'.format(model_clean_status,model_name, test_clean_status))
    outfile.close()
    print('--------------------'+ model_clean_status +' '+ model_name + ' on ' + test_clean_status + ' data --------------------')
    print_performance(model_name, label, y_predicted, model_clean_status, DATA_DIR, DATA)

    # Recording TPR and FPR for the TESTING ROC curves
    performance = {}
    performance['fpr'], performance['tpr'], thresh = h.roc_curve(label, y_predicted)
    performance['auc'] = h.roc_auc_score(label, y_predicted)
    print('Test AUC: %.3f' % (performance['auc']))
    return performance


"""train and test ML models. It takes in train and test data. Train_clean_status is either "clean" or "unclean"""
def train_test_ML_models(x_train, y_train, x_test, y_test, train_clean_status, DATA_DIR, DATA):
    """ Train ML models"""
    models = []
    models.append(
      ('DT', DecisionTreeClassifier(criterion='gini', min_samples_split=10, min_samples_leaf=5, max_features='auto')))
    models.append(('RF', RandomForestClassifier(max_depth=None, min_samples_split=10, min_samples_leaf=5,
                                                max_features='auto', bootstrap=True, verbose=0, criterion='gini')))

    report_scores_all = {}
    write_stats('model_name,train_status, macrof1\n', 'F1', DATA_DIR, DATA)
    write_stats('model_name,train_status, AUC\n', 'AUC', DATA_DIR, DATA)
    write_stats('model_name,train_status, FPR\n', 'FPR', DATA_DIR, DATA)
    write_stats('model_name,train_status, FNR\n', 'FNR', DATA_DIR, DATA)

    # Train and test model with clean data
    for model_name, model in models:
      print('Training %s model...' % model_name)
      # Train with clean data
      model.fit(x_train, y_train)

      # Save model
      #save_models(model, model_name)

      # Test with clean data
      report_scores = test_model(x_test, y_test, model, model_name, 'clean', train_clean_status, DATA_DIR, DATA)
      report_scores_all[model_name + '_{}'.format(train_clean_status)] = report_scores

import torch, gc
from GPUtil import showUtilization as gpu_usage

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()
    gc.collect()

    # del device
    torch.cuda.empty_cache()

    print("\nGPU Usage after emptying the cache")
    gpu_usage()




