import utils
import header as h

from pytorch_pretrained_bert import BertModel
from torch import nn
from pytorch_pretrained_bert import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import torch

#import pyprind
#import sys
import time

#import main

torch.cuda.empty_cache() # free cuda memory to avoid cuda out of mem

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from IPython.display import clear_output

import numpy as np
import os


SEQ_WINDOW = 25
#BATCH_SZ = 32
#EPOCHS = 2
CLEAN = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device in BERT: ', device)

# Define BERT model
class BertBinaryClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(BertBinaryClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tokens, masks=None):
        # First Layer
        _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)

        dropout_output = self.dropout(pooled_output)

        linear_output = self.linear(dropout_output)

        # output layer
        proba = self.sigmoid(linear_output)

        return proba

    def train_m(self, x, y, train_mask, epochs, batchsize):
        train_tokens_tensor = torch.tensor(x)
        train_y_tensor = torch.tensor(y.reshape(-1, 1)).float()
        train_masks_tensor = torch.tensor(train_mask)

        train_dataset = TensorDataset(train_tokens_tensor, train_masks_tensor, train_y_tensor)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batchsize)

        param_optimizer = list(self.sigmoid.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        optimizer = Adam(self.bert.parameters(), lr=2e-5)
        for epoch_num in range(epochs):
            self.train()  # Training Flag
            train_loss = 0

            #with tqdm(train_dataloader, unit="batch") as tepoch: #new

            for step_num, batch_data in enumerate(train_dataloader):
                # Load batch on device memory
                token_ids, masks, labels = tuple(t.to(device) for t in batch_data)

                # Get the output of the model for provided input
                logits = self(token_ids, masks)

                # Loss function
                loss_func = nn.BCELoss()

                # Calculate Loss
                batch_loss = loss_func(logits, labels)
                train_loss += batch_loss.item()

                # backpropagate the error
                self.zero_grad()
                batch_loss.backward()

                # Update the Weights of the Model
                clip_grad_norm_(parameters=self.parameters(), max_norm=1.0)
                optimizer.step()

                clear_output(wait=True)
                print('Epoch: ', epoch_num + 1, end="\r")
                print("\r" + "{0}/{1} loss: {2} ".format(step_num, len(y) / batchsize,
                                                         train_loss / (step_num + 1)))
           #     tepoch.set_postfix("{0}/{1} loss: {2} ".format(step_num, len(y) / batchsize,
            #                                             train_loss / (step_num + 1))) #(loss=loss.item(), accuracy=100. * accuracy)
                time.sleep(0.1)
            #    tepoch.close()

                    #sys.stdout.flush()


def process_text(data, label):
    texts = []
    for i in range(data.shape[0]):
        texts.append(" ".join(np.array(data.iloc[i, :]).astype(str)))
    texts = tuple(texts)

    label = tuple(label.tolist())

    return texts, label

def save_pretrained_model(model, model_name, directory):
    # Create folder to save model if the folder hasn't existed

    if not os.path.exists(directory + '/saved_models/'):
        os.makedirs(directory + '/saved_models/')

    if CLEAN is True:
        clean_status = 'clean'
    else:
        clean_status = 'unclean'

    # Save model
 #   torch.save(model, DATA_DIR + '/saved_models/{}{}'.format(model_name, clean_status))

    # Load model
#    bert_clf = torch.load(data_dir + '/saved_models/BERT{}'.format(clean_status))

def BERT_process_texts(x_train, y_train, test_clean, y_test_clean, test_unclean, y_test_unclean):

    train_texts, train_labels = process_text(x_train, y_train)
    clean_test_texts, clean_test_labels = process_text(test_clean, y_test_clean)
    unclean_test_texts, unclean_test_labels = process_text(test_unclean, y_test_unclean)

    # Convert labels to True/False
    train_y = np.array(train_labels) == 1
    test_clean_y = np.array(clean_test_labels) == 1
    test_unclean_y = np.array(unclean_test_labels) == 1

    # BERT Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Convert to tokens using tokenizer
    train_tokens = list(map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:SEQ_WINDOW] + ['[SEP]'], train_texts))
    clean_test_tokens = list(
        map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:SEQ_WINDOW] + ['[SEP]'], clean_test_texts))
    unclean_test_tokens = list(
        map(lambda t: ['[CLS]'] + tokenizer.tokenize(t)[:SEQ_WINDOW] + ['[SEP]'], unclean_test_texts))

    # Following is to convert List of words to list of numbers. (Words are replaced by their index in dictionar)
    train_tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, train_tokens)), maxlen=SEQ_WINDOW,
                                     truncating="post", padding="post", dtype="int")
    clean_test_tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, clean_test_tokens)),
                                          maxlen=SEQ_WINDOW, truncating="post", padding="post", dtype="int")
    unclean_test_tokens_ids = pad_sequences(list(map(tokenizer.convert_tokens_to_ids, unclean_test_tokens)),
                                            maxlen=SEQ_WINDOW, truncating="post", padding="post", dtype="int")

    # To mask the paddings
    train_masks = [[float(i > 0) for i in ii] for ii in train_tokens_ids]
    clean_test_masks = [[float(i > 0) for i in ii] for ii in clean_test_tokens_ids]
    unclean_test_masks = [[float(i > 0) for i in ii] for ii in unclean_test_tokens_ids]
    #-------------------------------------------------------
    # Convert token ids to tensor
    clean_test_tokens_tensor = torch.tensor(clean_test_tokens_ids)
    unclean_test_tokens_tensor = torch.tensor(unclean_test_tokens_ids)

    # Convert labels to tensors
    clean_test_y_tensor = torch.tensor(test_clean_y.reshape(-1, 1)).float()
    unclean_test_y_tensor = torch.tensor(test_unclean_y.reshape(-1, 1)).float()

    # Convert to tensor for maks
    clean_test_masks_tensor = torch.tensor(clean_test_masks)
    unclean_test_masks_tensor = torch.tensor(unclean_test_masks)

    # Load Token, token mask and label into Dataloader
    clean_test_dataset = TensorDataset(clean_test_tokens_tensor, clean_test_masks_tensor, clean_test_y_tensor)
    unclean_test_dataset = TensorDataset(unclean_test_tokens_tensor, unclean_test_masks_tensor, unclean_test_y_tensor)
    return train_tokens_ids, train_y, train_masks, clean_test_dataset, test_clean_y, unclean_test_dataset, test_unclean_y


def evaluate_Bert(bert_clf, dataloader):
    bert_clf.eval()  # Define eval
    bert_predicted = []  # To Store predicted result
    all_logits = []  # Actual output that is between 0 to 1 is stored here
    with torch.no_grad():
        for step_num, batch_data in enumerate(dataloader):
            # Load the batch on gpu memory
            token_ids, masks, labels = tuple(t.to(device) for t in batch_data)

            # Calculate ouput of bert
            logits = bert_clf(token_ids, masks)

            # Get the numpy logits
            numpy_logits = logits.cpu().detach().numpy()  # Detach from the GPU memory

            # Using the threshold find binary
            bert_predicted += list(numpy_logits[:, 0] > 0.5)  # Threshold conversion
            all_logits += list(numpy_logits[:, 0])
    return bert_predicted


def get_Bert_performance(bert_clf, dataloader, label, test_clean_status, train_clean_status, DATA_DIR, DATA):
    bert_predicted = evaluate_Bert(bert_clf, dataloader)

    outfile = open(DATA_DIR+'significance_test/{}-Output.txt'.format(DATA), "a")  # Live-Named-Output.txt
    outfile.write('---------------------Testing {} BERT with {} test data-------------------'.format(train_clean_status, test_clean_status))
    outfile.close()

    print('---------------------Testing {} BERT with {} test data-------------------'.format(train_clean_status, test_clean_status))
    utils.print_performance('BERT', label, bert_predicted, train_clean_status, DATA_DIR, DATA)
   #params: model_name, true_labels, pred_labels, model_clean_status, DATA_DIR, DATA
    # Recording TPR and FPR for the TESTING-ROC curves
    performance = {}
    performance['fpr'], performance['tpr'], thresh = h.roc_curve(label, bert_predicted)
    performance['auc'] = h.roc_auc_score(label, bert_predicted)
    return performance


def train_and_test_BERT(train_clean_status, x_train, y_train, test_clean, y_test_clean, test_unclean, y_test_unclean, DATA_DIR, DATA, BATCH_SZ, EPOCHS):
    # Set device to use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_clf = BertBinaryClassifier()
    bert_clf = bert_clf.cuda()

    train_tokens_ids, train_y, train_masks, clean_test_dataset, test_clean_y, unclean_test_dataset, \
    test_unclean_y = BERT_process_texts(x_train, y_train, test_clean, y_test_clean, test_unclean, y_test_unclean)

    # Train BERT NLP
    if train_clean_status == "unclean":
        EPOCHS = 2; BATCH_SZ = 32

    bert_clf.train_m(train_tokens_ids, train_y, train_masks, EPOCHS, BATCH_SZ)

    # Save model
  #  save_pretrained_model(bert_clf, 'BERT', main.data_dir)

    # Define sampler
    clean_test_sampler = SequentialSampler(clean_test_dataset)
    unclean_test_sampler = SequentialSampler(unclean_test_dataset)

    # Defile test data loader
    clean_test_dataloader = DataLoader(clean_test_dataset, sampler=clean_test_sampler, batch_size=128)
    unclean_test_dataloader = DataLoader(unclean_test_dataset, sampler=unclean_test_sampler, batch_size=128)

    print('----------------------------Evaluating BERT with Clean Data----------------------------')
    BERT_clean_perf = get_Bert_performance(bert_clf, clean_test_dataloader, test_clean_y, 0, train_clean_status, DATA_DIR, DATA)
    #params: bert_clf, dataloader, label, test_clean_status, train_clean_status, DATA_DIR, DATA

    #print('----------------------------Evaluating BERT with Unclean Data----------------------------')
    #BERT_unclean_perf = get_Bert_performance(unclean_test_dataloader, test_unclean_y, 101)