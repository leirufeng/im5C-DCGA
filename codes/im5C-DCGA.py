import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

from keras.layers import Input, Conv1D, GlobalAvgPool2D, GlobalAveragePooling1D, \
    Dropout, Dense, Activation, Concatenate, Multiply, MaxPool2D, Add, recurrent, \
    LSTM, Bidirectional, Conv2D, AveragePooling2D, BatchNormalization, Flatten, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Reshape, Permute, multiply, Lambda, add, subtract, MaxPooling2D, GRU, ReLU
from keras.regularizers import l1, l2
from keras.optimizer_v2.adam import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from keras.models import Model, load_model
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from keras.layers import Layer,GaussianNoise,Embedding
from keras import initializers
from random import randint, choice
from sklearn.metrics import precision_recall_curve, auc
import h5py

import warnings

# read DNA sequences
def read_fasta(fasta_file_name):
    seqs = []
    seqs_num = 0
    file = open(fasta_file_name)

    for line in file.readlines():
        if line.strip() == '':
            continue

        if line.startswith('>'):
            seqs_num = seqs_num + 1
            continue
        else:
            seq = line.strip()

            result1 = 'N' in seq
            result2 = 'n' in seq
            if result1 == False and result2 == False:
                seqs.append(seq)
    return seqs


# One-hot coding
def to_one_hot(seqs):
    base_dict = {
        'a': 0, 'c': 1, 'g': 2, 't': 3,
        'A': 0, 'C': 1, 'G': 2, 'T': 3
    }

    one_hot_4_seqs = []
    for seq in seqs:

        one_hot_matrix = np.zeros([4, len(seq)], dtype=float)
        index = 0
        for seq_base in seq:
            one_hot_matrix[base_dict[seq_base], index] = 1
            index = index + 1

        one_hot_4_seqs.append(one_hot_matrix)
    return one_hot_4_seqs



# NCP coding
def to_properties_code(seqs):
    properties_code_dict = {
        'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0], 'T': [0, 0, 1],
        'a': [1, 1, 1], 'c': [0, 1, 0], 'g': [1, 0, 0], 't': [0, 0, 1]
    }
    properties_code = []
    for seq in seqs:
        properties_matrix = np.zeros([3, len(seq)], dtype=float)
        m = 0
        for seq_base in seq:
            properties_matrix[:, m] = properties_code_dict[seq_base]
            m = m + 1
        properties_code.append(properties_matrix)
    return properties_code



# Performance evaluation
def show_performance(y_true, y_pred):

    TP, FP, FN, TN = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1:
            if y_pred[i] > 0.5:
                TP += 1
            else:
                FN += 1
        if y_true[i] == 0:
            if y_pred[i] > 0.5:
                FP += 1
            else:
                TN += 1


    Sn = TP / (TP + FN + 1e-06)

    Sp = TN / (FP + TN + 1e-06)

    Acc = (TP + TN) / len(y_true)

    MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + 1e-06)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    F1_score = 2 * (precision * recall) / (precision + recall)

    return Sn, Sp, Acc, MCC, F1_score


def performance_mean(performance):
    print('Sn = %.4f ± %.4f' % (np.mean(performance[:, 0]), np.std(performance[:, 0])))
    print('Sp = %.4f ± %.4f' % (np.mean(performance[:, 1]), np.std(performance[:, 1])))
    print('Acc = %.4f ± %.4f' % (np.mean(performance[:, 2]), np.std(performance[:, 2])))
    print('Mcc = %.4f ± %.4f' % (np.mean(performance[:, 3]), np.std(performance[:, 3])))
    print('Auc = %.4f ± %.4f' % (np.mean(performance[:, 4]), np.std(performance[:, 4])))
    print('F1_score = %.4f ± %.4f' % (np.mean(performance[:, 5]), np.std(performance[:, 5])))
    print('pr_auc = %.4f ± %.4f' % (np.mean(performance[:, 6]), np.std(performance[:, 6])))



if __name__ == '__main__':

    np.random.seed(0)
    tf.random.set_seed(1)

    # read data

   train_hep_pos_seqs = np.array(read_fasta('../Human hepatocellular carcinoma(HUH7_LIVER)/positive.fasta'))
    train_hep_neg_seqs = np.array(read_fasta('../Human hepatocellular carcinoma(HUH7_LIVER)/negative.fasta'))

    train_nsmall_pos_seqs = np.array(read_fasta('../Human non-small cell lung cancer (NCIH2228_LUNG)/positive.fasta'))
    train_nsmall_neg_seqs = np.array(read_fasta('../Human non-small cell lung cancer (NCIH2228_LUNG)/negative.fasta'))

    train_small_pos_seqs = np.array(read_fasta('../Human small cell lung cancer/all_positive.fasta'))
    train_small_neg_seqs = np.array(read_fasta('../Human small cell lung cancer/all_negative.fasta'))

    train_pos_seqs = np.concatenate((train_hep_pos_seqs, train_nsmall_pos_seqs, train_small_pos_seqs), axis=0)
    train_neg_seqs = np.concatenate((train_hep_neg_seqs, train_nsmall_neg_seqs, train_small_neg_seqs), axis=0)
    # train_seqs = np.concatenate((train_pos_seqs, train_neg_seqs), axis=0)




    train_pos_onehot = np.array(to_one_hot(train_pos_seqs)).astype(np.float32)
    train_pos_properties = np.array(to_properties_code(train_pos_seqs)).astype(np.float32)
    train_pos_all= np.concatenate((train_pos_onehot, train_pos_properties), axis=1)
    train_pos_label = np.array([1] * 386773 ).astype(np.float32)

    train_pos, test_pos, train_pos_label, test_pos_label = train_test_split(train_pos_all, train_pos_label, test_size=0.2, shuffle=True,
                                                            random_state=42)


    train_neg_onehot = np.array(to_one_hot(train_neg_seqs)).astype(np.float32)
    train_neg_properties = np.array(to_properties_code(train_neg_seqs)).astype(np.float32)
    train_neg_all = np.concatenate((train_neg_onehot, train_neg_properties), axis=1)
    train_neg_label = np.array([0] * 2923859).astype(np.float32)
    train_neg, test_neg, train_neg_label, test_neg_label = train_test_split(train_neg_all, train_neg_label, test_size=0.2, shuffle=True,
                                                            random_state=42)

    train = np.concatenate((train_pos, train_neg), axis=0)
    train_label = np.concatenate((train_pos_label, train_neg_label), axis=0)
    test = np.concatenate((test_pos, test_neg), axis=0)
    test_label = np.concatenate((test_pos_label, test_neg_label), axis=0)

    # build model

    model = build_model()


    BATCH_SIZE = 1024
    EPOCHS = 5
    weights = {0: 1, 1: 7.6}
    # Cross-validation
    n = 5
    k_fold = StratifiedKFold(n_splits=n, shuffle=True, random_state=42)

    all_performance = []

    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    for fold_count, (train_index, val_index) in enumerate(k_fold.split(train, train_label)):
        print('*' * 30 + ' the ' + str(fold_count + 1) + ' fold ' + '*' * 30)

        trains, val = train[train_index], train[val_index]
        trains_label, val_label = train_label[train_index], train_label[val_index]
        trains_label = to_categorical(trains_label, num_classes=2)
        val_label = to_categorical(val_label, num_classes=2)


       

        model = load_model('../models/model_fold' + str(fold_count+1) + '.h5',custom_objects={'AttLayer': AttLayer})

        val_pred = model.predict(val, verbose=1)


        # Sn, Sp, Acc, MCC, AUC, F1_score, pr_auc
        Sn, Sp, Acc, MCC, F1_score = show_performance(val_label[:, 1], val_pred[:, 1])
        AUC = roc_auc_score(val_label[:, 1], val_pred[:, 1])
        precision, recall, thresholds = precision_recall_curve(val_label[:, 1], val_pred[:, 1])
        pr_auc = auc(recall, precision)
        print('-----------------------------------------------val---------------------------------------')
        print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f, F1_score = %f, pr_auc = %f' % (Sn, Sp, Acc, MCC, AUC, F1_score, pr_auc))

        val_performance = [Sn, Sp, Acc, MCC, AUC, F1_score, pr_auc]
        all_performance.append(val_performance)
        pd.DataFrame(val_performance).to_csv('../pre_file/model_fold' + str(fold_count+1) + '.csv', index=False)

        '''Mapping the ROC'''
        fpr, tpr, thresholds = roc_curve(val_label[:, 1], val_pred[:, 1], pos_label=1)

        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        plt.plot(fpr, tpr, label='ROC fold {} (AUC={:.4f})'.format(str(fold_count + 1), AUC))

    all_performance = np.array(all_performance)

    val_performance_mean = performance_mean(all_performance)

    '''Mapping the ROC'''
    plt.plot([0, 1], [0, 1], '--', color='red')
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = np.mean(np.array(all_performance)[:, 4])

    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC=%0.4f)' % (mean_auc), lw=2, alpha=.8)

    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('../images/5fold_ROC_Curve.jpg', dpi=1200, bbox_inches='tight')
    plt.legend(loc='lower right')
    plt.show()

    train_label = to_categorical(train_label, num_classes=2)
    test_label = to_categorical(test_label, num_classes=2)
    
    
    # test
  

    model = load_model('../models/model_test.h5', custom_objects={'AttLayer': AttLayer})

    test_score = model.predict(test)

    # Sn, Sp, Acc, MCC, AUC, F1_score, pr_auc
    Sn, Sp, Acc, MCC, F1_score = show_performance(test_label[:, 1], test_score[:, 1])
    AUC = roc_auc_score(test_label[:, 1], test_score[:, 1])
    precision, recall, thresholds = precision_recall_curve(test_label[:, 1], test_score[:, 1])
    pr_auc = auc(recall, precision)
    print('-----------------------------------------------test---------------------------------------')
    print('Sn = %f, Sp = %f, Acc = %f, MCC = %f, AUC = %f, F1_score = %f, pr_auc = %f' % (Sn, Sp, Acc, MCC, AUC, F1_score, pr_auc))

    #
    '''Mapping the ROC'''
    plt.plot([0, 1], [0, 1], '--', color='red')
    test_fpr, test_tpr, thresholds = roc_curve(test_label[:, 1], test_score[:, 1], pos_label=1)

    plt.plot(test_fpr, test_tpr, color='b', label=r'test ROC (AUC=%0.4f)' % (AUC), lw=2, alpha=.8)

    plt.title('ROC Curve OF')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('../images/test_ROC_Curve.jpg', dpi=1200, bbox_inches='tight')
    plt.legend(loc='lower right')
    plt.show()


