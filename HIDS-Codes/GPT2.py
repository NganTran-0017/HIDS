import utils
import header as h

import numpy as np
import torch
from torch import nn
torch.cuda.empty_cache() # free cuda memory to avoid cuda out of mem

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from IPython.display import clear_output

from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

#from ipywidgets import IntProgress
tokenizer = GPT2Tokenizer.from_pretrained('microsoft/DialoGPT-small')

SEQ_WINDOW = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device in GPT-2: ', device)

class GPT2BinaryClassifier(nn.Module):
    def __init__(self, dropout=0.1):
        super(GPT2BinaryClassifier, self).__init__()
        self.gpt2 = GPT2ForSequenceClassification.from_pretrained('microsoft/DialoGPT-small')

    def train_m(self, x, y, train_mask, epochs, batchsize):
        train_tokens_tensor = torch.tensor(x)
        train_y_tensor = torch.tensor(y.reshape(-1, 1)).long()
        train_masks_tensor = torch.tensor(train_mask)

        train_dataset = TensorDataset(train_tokens_tensor, train_masks_tensor, train_y_tensor)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batchsize)

        # param_optimizer = list(self.gpt2.parameters())
        # optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        optimizer = Adam(self.gpt2.parameters(), lr=5e-5)
        for epoch_num in range(epochs):
            self.gpt2.train()  # Training Flag
            train_loss = 0
            for step_num, batch_data in enumerate(train_dataloader):
                # Load batch on device memory
                token_ids, masks, labels = tuple(t.to(device) for t in batch_data)
                self.zero_grad()

                # Get the output of the model for provided input
                outputs = self.gpt2(token_ids, attention_mask=masks, labels=labels)
                loss, logits = outputs[:2]
                # logits = self(token_ids, masks)

                # Total Loss
                train_loss += loss.item()

                # Backward pass the loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.gpt2.parameters(), 1.0)

                optimizer.step()
                logits = logits.detach().cpu().numpy()

                clear_output(wait=True)

                print('Epoch: ', epoch_num + 1, end="\r")
                print("\r" + "{0}/{1} loss: {2} ".format(step_num, len(y) / batchsize,
                                                         train_loss / (step_num + 1)))

def process_text(data, label):
    texts = []
    for i in range(data.shape[0]):
        texts.append(" ".join(np.array(data.iloc[i, :]).astype(str)))
    texts = tuple(texts)

    label = tuple(label.tolist())

    return texts, label


# Evaluate Model
def evaluate_GPT(gpt_clf, dataloader):
    gpt_clf.eval()  # Define eval
    gpt_predicted = []  # Store Result
    with torch.no_grad():
        for step_num, batch_data in enumerate(dataloader):
            token_ids, masks, labels = tuple(t.to(device) for t in batch_data)

            # ----------------------------------------------------------------
            outputs = gpt_clf.gpt2(token_ids, attention_mask=masks, labels=labels)
            loss, logits = outputs[:2]
            numpy_logits = logits.detach().cpu().numpy()
            # ----------------------------------------------------------------
            gpt_predicted += list(numpy_logits.argmax(axis=-1).flatten().tolist())
    return gpt_predicted


def get_GPT_performance(gpt_clf, dataloader, label, test_clean_status, train_clean_status, DATA_DIR, DATA):
    #params: gpt_clf, dataloader, label, test_clean_status, train_clean_status, DATA_DIR, DATA

    gpt_predicted = evaluate_GPT(gpt_clf, dataloader)
    outfile = open(DATA_DIR + 'significance_test/{}-Output.txt'.format(DATA), "a")  # Live-Named-Output.txt

    outfile.write('---------------------Testing {} GPT with {} test data-------------------'.format(train_clean_status, test_clean_status))
    outfile.close()

    print('---------------------Testing {} GPT with {} test data-------------------'.format(train_clean_status, test_clean_status))
    utils.print_performance('GPT', label, gpt_predicted, train_clean_status, DATA_DIR, DATA)

    # Recording TPR and FPR for the TESTING-ROC curves
    performance = {}
    performance['fpr'], performance['tpr'], thresh = h.roc_curve(label, gpt_predicted)
    performance['auc'] = h.roc_auc_score(label, gpt_predicted)
    return performance

def GPT_process_texts(x_train, y_train, test_clean, y_test_clean, test_unclean, y_test_unclean):

    train_texts, train_labels = process_text(x_train, y_train)
    clean_test_texts, clean_test_labels = process_text(test_clean, y_test_clean)
    unclean_test_texts, unclean_test_labels = process_text(test_unclean, y_test_unclean)

    # Prepare labels
    # Convert labels to True/False True if intrusion or False if normal
    train_y = np.array(train_labels) == 1
    test_clean_y = np.array(clean_test_labels) == 1
    test_unclean_y = np.array(unclean_test_labels) == 1

    # Padding sequences from the right to a max length of 20
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    train_tokens        = tokenizer(train_texts, return_tensors='pt', truncation=True, padding=True, max_length=SEQ_WINDOW)
    clean_test_tokens   = tokenizer(clean_test_texts, return_tensors='pt', truncation=True, padding=True,
                                  max_length=SEQ_WINDOW)
    unclean_test_tokens = tokenizer(unclean_test_texts, return_tensors='pt', truncation=True, padding=True,
                                    max_length=SEQ_WINDOW)

    # Following is to convert List of words to list of numbers. (Words are replaced by their index in dictionary)
    train_tokens_ids = train_tokens.input_ids
    clean_test_tokens_ids = clean_test_tokens.input_ids
    unclean_test_tokens_ids = unclean_test_tokens.input_ids

    train_masks = train_tokens.attention_mask
    clean_test_masks = clean_test_tokens.attention_mask
    unclean_test_masks = unclean_test_tokens.attention_mask

    #-------------------------------------------------------------------------
    # Copy construct from token ids
    clean_test_tokens_tensor = clean_test_tokens_ids.clone().detach()  # torch.tensor(clean_test_tokens_ids)
    unclean_test_tokens_tensor = unclean_test_tokens_ids.clone().detach()  # torch.tensor(unclean_test_tokens_ids)
    clean_test_y_tensor = torch.tensor(test_clean_y.reshape(-1, 1)).long()
    unclean_test_y_tensor = torch.tensor(test_unclean_y.reshape(-1, 1)).long()

    clean_test_masks_tensor = clean_test_masks.clone().detach()  # torch.tensor(clean_test_masks)
    unclean_test_masks_tensor = unclean_test_masks.clone().detach()  # torch.tensor(unclean_test_masks)

    clean_test_dataset = TensorDataset(clean_test_tokens_tensor, clean_test_masks_tensor, clean_test_y_tensor)
    unclean_test_dataset = TensorDataset(unclean_test_tokens_tensor, unclean_test_masks_tensor, unclean_test_y_tensor)

    return train_tokens_ids, train_y, train_masks, clean_test_dataset, test_clean_y, unclean_test_dataset, test_unclean_y



def train_and_test_GPT(train_clean_status, x_train, y_train, test_clean, y_test_clean, test_unclean, y_test_unclean, DATA_DIR, DATA, BATCH_SZ, EPOCHS):
    # Set device to use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt_clf = GPT2BinaryClassifier()
    gpt_clf = gpt_clf.cuda()

    train_tokens_ids, train_y, train_masks, clean_test_dataset, test_clean_y, unclean_test_dataset, \
    test_unclean_y = GPT_process_texts(x_train, y_train, test_clean, y_test_clean, test_unclean, y_test_unclean)

    # Train GPT
    if train_clean_status == "unclean":
        EPOCHS = 2; BATCH_SZ = 32

    # Configure the Padding token id
    gpt_clf.gpt2.config.pad_token_id = tokenizer.eos_token_id
    gpt_clf.train_m(train_tokens_ids, train_y, train_masks, EPOCHS, BATCH_SZ)

    # Save model
  #  save_pretrained_model(gpt_clf, 'BERT', main.data_dir)

    # Define sampler
    clean_test_sampler = SequentialSampler(clean_test_dataset)
    unclean_test_sampler = SequentialSampler(unclean_test_dataset)

    clean_test_dataloader = DataLoader(clean_test_dataset, sampler=clean_test_sampler, batch_size=128)
    unclean_test_dataloader = DataLoader(unclean_test_dataset, sampler=unclean_test_sampler, batch_size=128)

    print('----------------------------Evaluating GPT with Clean Data----------------------------')
    GPT_clean_perf = get_GPT_performance(gpt_clf, clean_test_dataloader, test_clean_y, 0, train_clean_status, DATA_DIR, DATA)

    #params: gpt_clf, dataloader, label, test_clean_status, train_clean_status, DATA_DIR, DATA

   # print('----------------------------Evaluating GPT with Unclean Data----------------------------')
    #GPT_unclean_perf = get_GPT_performance(gpt_clf, unclean_test_dataloader, test_unclean_y, 0, train_clean_status, DATA_DIR, DATA)
