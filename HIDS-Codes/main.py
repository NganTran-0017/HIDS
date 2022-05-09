# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

"""(This project process data and partition them into training and testing sets.
    Then it cleans the testing data of overlaps and duplicates.
    If the option is to train w clean data, the program will clean the data.
        Otherwise, the training data is left as is.
    Save the training and testing data, just in case.
    Bootstrapping training data to create a balanced dataset
    Train and test DT, RF, BERT, GPT and record Macro F1, AUC, FPR and FNR in a text file with clean/unclean model.)

    -->This process is repeated 10 time for clean data, and then for unclean data
    Test significance in each metric once we get 10 results from and 10 results from unclean model
    """
import pandas as pd
import glob

import header as h
import utils
import BERT
import GPT2

# Bootstrapping size
SZ = 1

#def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    #print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

"""This func read in normal data and intrusion data separately given the data direction from gtihub in computer"""
def read_data():
    normal_files = glob.glob(DIR+GITHUB_DATA+'/*.txt') # or ..int file
    intrusion_files = h.os.listdir(DIR+GITHUB_DATA+'/intrusion/')
    print('Normal Files to read:', normal_files, '\nIntrusion files:', intrusion_files)
    normal, intrusion = h.pd.DataFrame(), h.pd.DataFrame()

    for f in normal_files:
        normal = h.pd.concat([normal, pd.read_csv( f, sep = ' ', header = None)])

    for f in intrusion_files:
        intrusion = h.pd.concat([intrusion, pd.read_csv(DIR+GITHUB_DATA + '/intrusion/'+f, sep = ' ', header = None)])

    normal = normal.iloc[:, :-2] # drop the last 2 columns, which only have NaN values
    normal = normal.rename(columns={0: "PID", 1: "Syscall"})

    intrusion = intrusion.iloc[:, :-2] # drop the last 2 columns, which only have NaN values
    intrusion = intrusion.rename(columns={0: "PID", 1: "Syscall"})

    print('Normal shape:', normal.shape, ' Intrusion shape: ', intrusion.shape)
    return normal, intrusion

def partition_and_process_train_test(normal, intrusion):
    # Pre-process data
    unclean_train, unclean_test = utils.preprocess_data(normal, intrusion)
    print('train shape: ', unclean_train.shape);
    print('test shape: ',  unclean_test.shape)
    print('train: ',       unclean_train.head(5))

    # Remove overlap between train and test from test set. Also remove duplicates from test set.
    # x_test is the unclean test and clean_test is clean
    clean_test = utils.clean_test(unclean_train, unclean_test)

    # Clean overlaps and duplicates between two classes. Clean test still has labels in it
    clean_test = utils.clean_data_between_two_classes(clean_test)
    print('Unclean test: ', unclean_test.shape, ' vs. Clean test: ', clean_test.shape)

    # Clean training data and store it in a new var. Clean train still has labels in it
    clean_train = utils.clean_data_between_two_classes(unclean_train)

    return unclean_train, clean_train, unclean_test, clean_test

"""Bootstrapping data to create a balanced dataset. 
This version is different from HIDS-project. """
def bootstrap_data(data):
    normal_class, intrusion_class = utils.separate_two_classes(data)
    print('Before bootstraping: Normal sz: %d vs Intrusion sz: %d' %(normal_class.shape[0], intrusion_class.shape[0]))

    #data['Label'] = label
    if len(intrusion_class) > len(normal_class):
        data = h.pd.concat([data.iloc[intrusion_class.index],
                                 data.iloc[normal_class.index].sample(n=len(intrusion_class), replace=True)])
    else:
        data = h.pd.concat(
            [data.iloc[normal_class.index], data.iloc[intrusion_class.index].sample(n=len(normal_class), replace=True)])

    data = data.sample(frac=float(SZ));
    data.reset_index(drop=True, inplace=True)  # Shuffle data with SZ fraction and reset index
    label = data['Label']
    data.drop(columns='Label', inplace=True)
    return data, label

DIR = 'C:/Users/natal/Documents/GitHub/Datasets/UNM/'
GITHUB_DATA = input('Enter Dataset folder from Github: ')
DATA = input('Enter Dataset for significance test: ')
# This batchsz and epochs will be used when trained on clean data. For unclean data, it will be automatically 32 batch sz and 2 eps
BATCH_SZ = int(input('Enter batch sz for clean data: '))
EPOCHS = int(input('Enter epochs num for clean data: '))

DATA_DIR = '../Processed-DATA/' + DATA + '/'      #input('Enter Dataset for significance test: ')
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Ls Data directory
    print('ls from ', GITHUB_DATA, ': ', h.os.listdir(DIR+GITHUB_DATA+'/'))

    # Read in data
    normal, intrusion = read_data()
    print(normal.columns, normal)
    print(intrusion)

    # Get model performances when trained w clean data vs. when trained with unclean data 10 times to run significance test
    for count in range(10):
        print('------------------------------Run #{}---------------------------------'.format(count))
        # Partition data into train and test, then clean them and store them separately
        unclean_train, clean_train, unclean_test, clean_test = partition_and_process_train_test(normal, intrusion)

        # Bootstrap Training Data. clean_x_train does not contain labels
        clean_x_train, clean_y_train     = bootstrap_data(clean_train)
        unclean_x_train, unclean_y_train = bootstrap_data(unclean_train)

        # Train ML models with clean data and test with clean data.
        utils.train_test_ML_models(clean_x_train, clean_y_train, clean_test.iloc[:, :-1], clean_test['Label'], 'clean', DATA_DIR, DATA)

        # Train pre-trained models with clean data and test with clean data
        #   clean up GPU before training
        utils.free_gpu_cache()
        print('Training and Testing BERT')
        BERT.train_and_test_BERT('clean',clean_x_train, clean_y_train, clean_test.iloc[:, :-1], clean_test['Label'],
                                 unclean_test.iloc[:, :-1], unclean_test['Label'], DATA_DIR, DATA, BATCH_SZ, EPOCHS)

        utils.free_gpu_cache()
        print('Training and Testing GPT')
        GPT2.train_and_test_GPT('clean', clean_x_train, clean_y_train, clean_test.iloc[:, :-1], clean_test['Label'],
                                 unclean_test.iloc[:, :-1], unclean_test['Label'], DATA_DIR, DATA, BATCH_SZ, EPOCHS)

        # Then, train ML with unclean data and test with clean data
        utils.train_test_ML_models(unclean_x_train, unclean_y_train, clean_test.iloc[:, :-1], clean_test['Label'], 'unclean',DATA_DIR, DATA)

        # Train pre-trained models with unclean data and test with clean data
        #   clean up GPU before training
        utils.free_gpu_cache()
        print('Training and Testing BERT')
        BERT.train_and_test_BERT('unclean', unclean_x_train, unclean_y_train, clean_test.iloc[:, :-1], clean_test['Label'],
                                 unclean_test.iloc[:, :-1], unclean_test['Label'], DATA_DIR, DATA, BATCH_SZ, EPOCHS)

        utils.free_gpu_cache()
        print('Training and Testing GPT')
        GPT2.train_and_test_GPT('unclean', unclean_x_train, unclean_y_train, clean_test.iloc[:, :-1], clean_test['Label'],
                                 unclean_test.iloc[:, :-1], unclean_test['Label'], DATA_DIR, DATA, BATCH_SZ, EPOCHS)
