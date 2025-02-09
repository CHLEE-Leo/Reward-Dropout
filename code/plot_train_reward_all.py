# %%
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import glob

# %%
parser = argparse.ArgumentParser(description='receive the parameters')
parser.add_argument('--dataset', type = str, required = True)       # dataset = {   'sentiment-0', 'sentiment-1', 
                                                                    #               'toxicity-0', 'toxicity-1', 
                                                                    #               'politeness-0', 'politeness-1',
                                                                    #               'emotion-0', 'emotion-1', 'emotion-2', 'emotion-3', 'emotion-4', 'emotion-5', 'emotion-6',
                                                                    #               'topic-0', 'act-1', 'act-2', 'act-3'}
parser.add_argument('--decoding', type = str, required = False)     # decoding : {'greedy', 'stochastic', 'top-k', 'top-p'}
parser.add_argument('--history', type = str, required = False)      # history : {'reward', 'acc', 'loss'}
args = parser.parse_args()
my_dataset = args.dataset
my_decoding = args.decoding
my_history = args.history + '_history'
# my_dataset = 'sentiment-0'
# my_decoding = 'top-k'


'''
데이터 주소 로드
'''
parent_dir = str(Path(os.getcwd()).parents[0])
RESULT_DIR = parent_dir + '/results'
REWARD_FILE_DIR_LIST = glob.glob(RESULT_DIR + '/{}/gpt2_small/*{}*[!.pdf]*'.format(my_dataset, my_decoding))

'''
데이터 로드
'''
for idx, each_dir in enumerate(REWARD_FILE_DIR_LIST):
    if idx == 0:
        my_dropout = each_dir.split('_')[-2]
        my_dropout_rate = each_dir.split('_')[-1]
        
        train_history_pd_all = pd.read_csv(each_dir + '/' + my_history + '.csv', index_col=0)
        train_history_pd_all['dropout'] = my_dropout
        train_history_pd_all['dropout_rate'] = my_dropout_rate
        print('dropout :', my_dropout)
        print('dropout_rate :', my_dropout_rate)
        print('length : ', train_history_pd_all.shape)

    else:
        my_dropout = each_dir.split('_')[-2]
        my_dropout_rate = each_dir.split('_')[-1]

        train_reward_pd = pd.read_csv(each_dir + '/' + my_history + '.csv', index_col=0)
        train_reward_pd['dropout'] = my_dropout
        train_reward_pd['dropout_rate'] = my_dropout_rate
        print('dropout :', my_dropout)
        print('dropout_rate :', my_dropout_rate)
        print('length : ', train_reward_pd.shape)

        train_history_pd_all = pd.concat([train_history_pd_all, train_reward_pd], axis=0)

'''
플롯팅
'''
dropout_list = np.unique(train_history_pd_all['dropout'])
dropout_rate_list = np.unique(train_history_pd_all['dropout_rate'])
plt.figure(figsize = (3, 3))
for i in dropout_list:

    if i == 'quantile':
        target_marker = 'o'
    elif i == 'random':
        target_marker = 'x'
    elif i == 'None':
        target_marker = 's'

    if i == 'None':
        j = 0.0
        first_filter = train_history_pd_all[train_history_pd_all['dropout'] == i]
        train_reward = first_filter['train_reward']

        plt.plot(train_reward, marker=target_marker, markersize=7.5, label='{}_{}'.format(i, j), alpha=0.5)
        
    else:
        for j in dropout_rate_list:
            if j != str(0.0):
                first_filter = train_history_pd_all[train_history_pd_all['dropout'] == i]
                second_filter = first_filter[first_filter['dropout_rate'] == j]
                train_reward = second_filter['train_reward']

                plt.plot(train_reward, marker=target_marker, markersize=5, label='{}_{}'.format(i, j))

plt.tight_layout()
plt.grid(True)
plt.xticks(fontsize=7, ticks=np.arange(0, 20, 1), labels=np.arange(1, 21, 1))
plt.xlabel('epoch')
plt.ylabel('mean reward')
plt.legend(bbox_to_anchor = (1.7, 1.0), loc='upper right')
plt.title('decoding: {}'.format(my_decoding))
# plt.title('dataset : {} - decoding : {}'.format(my_dataset, my_decoding))
# plt.show()
plt.savefig(RESULT_DIR + '/{}/gpt2_small'.format(my_dataset) + '/plot_{}_{}_{}.pdf'.format(my_dataset, my_decoding, my_history), bbox_inches='tight')