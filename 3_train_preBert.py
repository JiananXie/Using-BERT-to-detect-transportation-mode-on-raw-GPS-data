import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from Bert import Dataset, BertClassifier, label_dict_shl, label_dict_gl
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, cohen_kappa_score
from torch.nn.functional import nll_loss,softmax

def training(model, train_data, val_data, learning_rate, epochs):
    train, val = Dataset(train_data), Dataset(val_data)
    train_dataloader = DataLoader(train, batch_size=2, sampler=RandomSampler(train))
    val_dataloader = DataLoader(val, batch_size=2, sampler=SequentialSampler(val))

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_step, num_training_steps=total_step)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.to(device)  # Move criterion to device

    min_valid_loss = float('inf')
    patience_counter = 0
    patience_limit = 2
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        model.train()  # Set model to training mode

        for train_input, train_label in train_dataloader:
            train_label = train_label.type(torch.LongTensor).to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            optimizer.zero_grad()  # Zero the gradients before backward pass
            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item() 
            total_acc_train += acc

            batch_loss.backward()
            optimizer.step()
            scheduler.step()

        total_acc_val = 0
        total_loss_val = 0
        model.eval()  # Set model to evaluation mode

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                val_label = val_label.type(torch.LongTensor).to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()  
                total_acc_val += acc

        train_losses.append(total_loss_train / len(train_data))
        val_losses.append(total_loss_val / len(val_data))
        train_accs.append(total_acc_train / len(train_data))
        val_accs.append(total_acc_val / len(val_data))
        
        print(
            f'''Epochs: {epoch + 1} 
            | Train Loss: {total_loss_train / len(train_data): .3f} 
            | Train Accuracy: {total_acc_train / len(train_data): .3f} 
            | Val Loss: {total_loss_val / len(val_data): .3f} 
            | Val Accuracy: {total_acc_val / len(val_data): .3f}''')
        if total_loss_val < min_valid_loss:
            min_valid_loss = total_loss_val
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience_limit:
            print('Early Stopping')
            break

    fig, ax1 = plt.subplots()   
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(train_losses, color=color, label='Train Loss')
    ax1.plot(val_losses, color=color, linestyle='dashed', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlim([0, len(train_losses)])
    ax1.set_ylim([0, 1.5])
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(train_accs, color=color, label='Train Acc')
    ax2.plot(val_accs, color=color, linestyle='dashed', label='Val Acc')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0,1])
    ax2.legend(loc='upper right')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('train_val_loss_acc_pre.png')

def evaluate(model, test_data):
    model.eval()
    test = Dataset(test_data)
    total_acc_test = 0
    #Judge wether to use GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')


    if use_cuda:
        model = model.cuda()

    test_dataloader = DataLoader(test, batch_size=16, sampler=SequentialSampler(test))

    preds = []
    labels = []
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.type(torch.LongTensor).to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)


            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

            preds.extend(softmax(output,dim=1).cpu().numpy())
            labels.extend(test_label.cpu().numpy())
    # compute nll, f1, kappa, roc, cm    
    nll = nll_loss(torch.log(torch.tensor(preds)), torch.tensor(labels))

    num_classes = len(label_dict_gl)
    # Binarize the labels
    labels_binary = label_binarize(labels, classes=[i for i in range(num_classes)])
    preds = np.array(preds)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_binary[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label='{0} (area = {1:0.2f})'.format(label_dict_gl[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_pre.png')

    preds = np.argmax(preds,axis=1)
    f1 = f1_score(labels, preds, average='weighted')
    kappa = cohen_kappa_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d',cmap='Blues',xticklabels=label_dict_gl.values(),yticklabels=label_dict_gl.values())
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.savefig('cm_pre.png')

    print(
        f'''Test Accuracy: {total_acc_test / len(test_data): .3f},
        F1-score: {f1:.3f},
        NLL: {nll:.3f},
        CK: {kappa:.3f}''')


if __name__ == '__main__':

    #load data
    data_train = pd.read_csv('data_train_full_0.10_ws32.csv')
    data_valid = pd.read_csv('data_valid_full_0.10_ws32.csv')
    data_test = pd.read_csv('data_test_full_0.10_ws32.csv')

    model = BertClassifier()
    lr = 0.00001 #1e-5
    epochs = 10
    training(model, data_train, data_valid, lr, epochs)
    evaluate(model, data_test)
    
