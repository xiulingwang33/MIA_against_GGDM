import logging
import pickle
import os
import copy as cp

from easydict import EasyDict as edict
import numpy as np
import torch

from evaluation.stats import eval_torch_batch, adjs_to_graphs, eval_graph_list
from utils.arg_helper import *
from utils.graph_utils import *
from utils.loading_utils_mia import *
from utils.visual_utils import *
import itertools

from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import pandas as pd


if __name__ == "__main__":
    data_='enzymes'


    num_seeds = 3
    # config_dict.num_client = num_client

    step=10###to distinguish the most efficient epoch window in terms of gradient/loss

    if data_=='mutag':
        graph_s_gen_dirs_s='XXX'


        num_class=2


    elif data_=='imdb-binary':
        graph_s_gen_dirs_s='XXX'

        num_class = 2

    elif data_=='imdb-multi':
        graph_s_gen_dirs_s='XXX'

        num_class = 3

    elif data_=='enzymes':
        graph_s_gen_dirs_s='XXX'

        num_class = 6

    # mem=[]
    # non_mem=[]
    result_all=[]
    for jj in range(num_class):

        for ii in range(0,num_seeds):

            #####add your own gradient/loss file path

            with open('./exp/{}/mia_{}_{}_{}_gradient2.csv'.format(data_, ii,
                                                                                                      jj,
                                                                                                      ii),
                    'rb') as f:
                loss_tuples = pickle.load(f)

            print(len(loss_tuples))

            mem0 = []
            for loss_ in loss_tuples:
                mem0.append(np.array(loss_[0].detach().cpu()))
            # mem0 = np.array(loss_tuples[0].detach().cpu())

            # mem = np.array(loss_tuples[1])[:, 2]
            # mem=np.tile(mem, (2,1)).T
            # print(mem0,np.shape(mem0))
            mem0 = np.array(mem0)

            non_mem = []

            for kk in range(0,num_seeds):
                if ii==kk:
                    # mem = np.array(loss_tuples[1])[:, 2]
                    # mem=np.tile(mem, (2,1)).T
                    continue
                    # print('***mem', jj, ii, kk, loss_tuples[1])
                else:

                    with open('./exp/{}/mia_{}_{}_{}_gradient2.csv'.format(data_, ii, jj,
                                                                                                              kk),
                            'rb') as f:
                        loss_tuples=pickle.load(f)

                    # print(loss_tuples[1])
                    # print(jj,ii,kk,np.shape(loss_tuples[1]))
                    # non_mem0=np.array(loss_tuples[1])
                    non_mem0 = []
                    for loss_ in loss_tuples:
                        non_mem0.append(np.array(loss_[0].detach().cpu()))

                    non_mem0 = np.array(non_mem0)
                    print(np.shape(non_mem0))
                    # exit()

                # non_mem = np.array(loss_tuples[1])[:, 2]
                    # non_mem = np.tile(non_mem, (2, 1)).T
                    # print('***non_mem', jj, ii, kk, loss_tuples[1])


                for l1 in range(0,np.shape(mem0)[1]-5,5):
                    # print(np.shape(mem)[1])
                    for l2 in range(l1+5,np.shape(mem0)[1],5):

                        mem_=cp.deepcopy(mem0)
                        non_mem_ = cp.deepcopy(non_mem0)

                        mem_=mem_[:,l1:l2]
                        non_mem_ =non_mem_[:,l1:l2]

                        mem_=np.array(mem_)
                        non_mem_=np.array(non_mem_)

                        # mem_=np.array(list(itertools.chain.from_iterable(mem)))
                        # non_mem_=np.array(list(itertools.chain.from_iterable(non_mem)))
                        # non_mem_ = np.array(list(itertools.chain.from_iterable(non_mem)))
                        # non_mem_ = np.array(list(itertools.chain.from_iterable(non_mem)))

                        test_len=min(int(np.shape(mem_)[0]),int(np.shape(non_mem_)[0]))
                        mem=mem_[0:test_len]
                        non_mem=non_mem_[0:test_len]

                        print('***',np.shape(mem),np.shape(non_mem))

                        label_mem = np.ones((np.shape(mem)[0], np.shape(mem)[1]))
                        label_non_mem = np.zeros((np.shape(mem)[0], np.shape(mem)[1]))
                        print(np.shape(mem),np.shape(non_mem))
                        # print(mem,non_mem)

                        from sklearn.model_selection import train_test_split

                        X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(mem,
                                                                                                    label_mem,
                                                                                                    test_size=0.3,
                                                                                                    random_state=42)

                        X_test_train, X_test_test, y_test_train, y_test_test = train_test_split(non_mem, label_non_mem,
                                                                                                test_size=0.3, random_state=42)

                        X_train = np.concatenate((X_train_train, X_test_train), axis=0)
                        X_test = np.concatenate((X_train_test, X_test_test), axis=0)
                        y_train = np.concatenate((y_train_train, y_test_train), axis=0)
                        y_test = np.concatenate((y_train_test, y_test_test), axis=0)

                        from sklearn import metrics
                        from sklearn.neural_network import MLPClassifier

                        mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), random_state=1,
                                            max_iter=500)

                        mlp.fit(X_train, y_train[:, 1])

                        print("Training set score: %f" % mlp.score(X_train, y_train[:, 1]))
                        print("Test set score: %f" % mlp.score(X_test, y_test[:, 1]))

                        with open('/exp/{}/mia_{}_{}_{}_{}_{}_mlp-gradient.pkl'.format(data_, ii, jj,kk,l1,l2), 'wb') as f:
                            pickle.dump(mlp, f)

                        y_score = mlp.predict(X_test)
                        proba = mlp.predict_proba(X_test)
                        # proba = np.amax(proba, axis=1)
                        proba = proba[:, 1]
                        print(metrics.f1_score(y_test[:, 1], y_score, average='micro'))
                        print(metrics.classification_report(y_test[:, 1], y_score, labels=range(3)))
                        # report = metrics.classification_report(y_test[:, 1], y_score, labels=range(3), output_dict=True)

                        # out="{}{}-mlp_sim-report.csv".format(res_dir, F)
                        #
                        # df = pd.DataFrame(report).transpose()
                        # df.to_csv(out, index=True)

                        acc = accuracy_score(y_test[:, 1], y_score)
                        recall = recall_score(y_score, y_test[:, 1])
                        precision = precision_score(y_score, y_test[:, 1])
                        f1 = f1_score(y_score, y_test[:, 1])
                        auc = roc_auc_score(y_test[:, 1], proba)

                        print(ii,kk, jj, l1,l2,acc, recall, precision, f1, auc)

                        # tsts = []
                        # for i in range(len(y_score)):
                        #     tst = [y_score[i], proba[i], y_test[i][1],l1,l2]
                        #     tsts.append(tst)
                        # name = ['y_score', 'y_prob', 'y_test_grd','begin','end']
                        # result = pd.DataFrame(columns=name, data=tsts)
                        # result.to_csv(
                        #     res+'/exp/{}/mia_{}_{}_{}_mlp1-gradient.csv'.format(data_, ii, jj,kk))
                        result_all.append([data_, ii, kk,jj,l1,l2, acc, recall, precision, f1, auc])

                        # exit()


                name = ['data', 'client1','client2', 'class','begin','end', 'acc', 'recall', 'precision', 'f1', 'auc']
                result = pd.DataFrame(columns=name, data=result_all)
                result.to_csv('/exp/mia_all_{}-gradient.csv'.format(data_))

                    # exit()

