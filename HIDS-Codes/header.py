import pandas as pd
import numpy as np

from scipy import stats
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, roc_curve, auc, recall_score, precision_score,plot_confusion_matrix
from sklearn.model_selection import train_test_split
import os

import nltk
#nltk.download("popular")

from nltk.tokenize import word_tokenize
from nltk import ngrams