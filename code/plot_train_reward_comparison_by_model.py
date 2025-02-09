# %%
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import glob
import copy

parser = argparse.ArgumentParser(description='receive the parameters')
parser.add_argument('--my_seed', type = int, required = True)
parser.add_argument('--dataset', type = str, required = True)       # dataset = {   'sentiment-0', 'sentiment-1', 
                                                                    #               'toxicity-0', 'toxicity-1', 
                                                                    #               'politeness-0', 'politeness-1',
                                                                    #               'emotion-0', 'emotion-1', 'emotion-2', 'emotion-3', 'emotion-4', 'emotion-5', 'emotion-6',
                                                                    #               'topic-0', 'act-1', 'act-2', 'act-3'}
parser.add_argument('--plot_model', type = str, required = False)        # plot_model : {'all', 'gpt2_small', 'opt', 'xglm', 'gpt2_large'}
parser.add_argument('--history', type = str, required = False)      # history : {'reward', 'acc', 'loss'}
args = parser.parse_args()
my_dataset = args.dataset
# my_dataset = 'topic-3'
my_history = args.history + '_history'
# my_history = 'reward_history'
my_plot_model = args.plot_model

'''
모델 별 사이즈
'''
if my_plot_model == 'gpt2_small':
    my_model_size = '117M'
elif my_plot_model == 'opt':
    my_model_size = '350M'
elif my_plot_model == 'xglm':
    my_model_size = '564M'
elif my_plot_model == 'gpt2_large':
    my_model_size = '774M'
'''
데이터 주소 로드
'''
parent_dir = str(Path(os.getcwd()).parents[0])
RESULT_DIR = parent_dir + '/results'
REWARD_FILE_DIR_LIST = []

# 모든 model의 결과를 하나의 플롯에 뽑는다면
if my_plot_model == 'all':
    model_list = ['gpt2_small', 'opt', 'xglm', 'gpt2_large']

# 각 model의 결과를 각 플롯에 뽑는다면
else:
    model_list = [my_plot_model]
    model_size = my_model_size

for my_model in model_list:
    REWARD_FILE_DIR_LIST += glob.glob(RESULT_DIR + '/{}/{}/*{}*{}*[!.pdf]*'.format(my_dataset, my_model, 'stochastic', 'quantile'))

'''
데이터 로드
'''
for idx, each_dir in enumerate(REWARD_FILE_DIR_LIST):

    if each_dir.split('_')[-1] == '0.85':
        pass;
    else:
        my_model = each_dir.split('/')[8]                   # e.g., gpt2_small
        my_dropout = each_dir.split('_')[-2]                # e.g., quantile
        my_dropout_rate = each_dir.split('_')[-1]           # e.g., 0.95
        
        train_history_pd = pd.read_csv(each_dir + '/' + my_history + '.csv', index_col=0)           # e.g., 
        train_history_pd['model'] = my_model
        train_history_pd['dropout_rate'] = my_dropout_rate
        print('model :', my_model)
        print('dropout_rate :', my_dropout_rate)
        print('length : ', train_history_pd.shape)

        if train_history_pd.shape[0] > 5:
            train_history_pd = train_history_pd.iloc[:5, :]

        if idx == 0:
            train_history_pd_all = copy.deepcopy(train_history_pd)

        else:
            train_history_pd_all = pd.concat([train_history_pd_all, train_history_pd], axis=0)

    
'''
플롯팅
'''
dropout_rate_list = np.unique(train_history_pd_all['dropout_rate'])
plt.figure(figsize = (2.5, 2.5))
if my_plot_model == 'all':
    for my_model in model_list:
        for my_dropout_rate in dropout_rate_list:

            if my_model == 'gpt2_small':
                target_marker = '+'
            elif my_model == 'opt':
                target_marker = '.'
            elif my_model == 'xglm':
                target_marker = 'x'
            elif my_model == 'gpt2_large':
                target_marker = "*"

            first_filter = train_history_pd_all[train_history_pd_all['model'] == my_model]
            second_filter = first_filter[first_filter['dropout_rate'] == my_dropout_rate]
            target_history = second_filter['train_{}'.format(args.history)]
            # target_history = second_filter['train_{}'.format('reward')]

            # if my_model == 'gpt2_small':
            #     plt.plot(target_history, marker=target_marker, markersize=10, label='{}_{}'.format(my_model.replace('_small', ''), my_dropout_rate))
            # elif my_model == 'gpt2_large':
            #     plt.plot(target_history, marker=target_marker, markersize=10, label='{}_{}'.format(my_model.replace('_large', ''), my_dropout_rate))
            # else:
            #     plt.plot(target_history, marker=target_marker, markersize=10, label='{}_{}'.format(my_model, my_dropout_rate))

            plt.plot(target_history, marker=target_marker, markersize=10, label='{}_{}'.format('quantile', my_dropout_rate))


    plt.tight_layout()
    plt.grid(True)
    plt.xticks(fontsize=10, ticks=np.arange(0, 5, 1), labels=np.arange(1, 6, 1))
    plt.xlabel('epoch')

    plt.ylabel('mean {}'.format(args.history))
    # plt.ylabel('mean {}'.format('reward'))

    # plt.legend(bbox_to_anchor = (1.0, 1.0), loc='upper left')
    plt.title('model : {}\ndecoding: {}\ndropout : {}'.format(my_plot_model, 'stochastic', 'quantile'))
    plt.show()
    plt.savefig(RESULT_DIR + '/{}'.format(my_dataset) + '/{}_comparison_5_epoch_plot_{}_{}_{}_{}.pdf'.format(my_plot_model, my_dataset, 'stochastic', 'quantile', args.history), bbox_inches='tight')

else:
    target_marker = '.'
    for my_model in model_list:
        for my_dropout_rate in dropout_rate_list:

            first_filter = train_history_pd_all[train_history_pd_all['model'] == my_model]
            second_filter = first_filter[first_filter['dropout_rate'] == my_dropout_rate]
            target_history = second_filter['train_{}'.format(args.history)]
            # target_history = second_filter['train_{}'.format('reward')]

            # if my_model == 'gpt2_small':
            #     plt.plot(target_history, marker=target_marker, markersize=10, label='{}_{}'.format(my_model.replace('_small', ''), my_dropout_rate))
            # elif my_model == 'gpt2_large':
            #     plt.plot(target_history, marker=target_marker, markersize=10, label='{}_{}'.format(my_model.replace('_large', ''), my_dropout_rate))
            # else:
            #     plt.plot(target_history, marker=target_marker, markersize=10, label='{}_{}'.format(my_model, my_dropout_rate))

            plt.plot(target_history, marker=target_marker, markersize=10, label='{}_{}'.format('quantile', my_dropout_rate))


    plt.tight_layout()
    plt.grid(True)
    plt.xticks(fontsize=10, ticks=np.arange(0, 5, 1), labels=np.arange(1, 6, 1))
    plt.xlabel('epoch')

    plt.ylabel('mean {}'.format(args.history))
    # plt.ylabel('mean {}'.format('reward'))

    if my_plot_model == 'gpt2_large':
        # plt.title('model : {} ({})\ndecoding: {}'.format(my_plot_model.replace('_large', ''), model_size, 'stochastic'))
        plt.title('model : {} ({})'.format(my_plot_model.replace('_large', ''), model_size))

    elif my_plot_model == 'gpt2_small':
        # plt.title('model : {} ({})\ndecoding: {}'.format(my_plot_model.replace('_small', ''), model_size, 'stochastic'))
        plt.title('model : {} ({})'.format(my_plot_model.replace('_small', ''), model_size))

    else:
        # plt.title('model : {} ({})\ndecoding: {}'.format(my_plot_model, model_size, 'stochastic'))
        plt.title('model : {} ({})'.format(my_plot_model, model_size))

    plt.show()
    plt.savefig(RESULT_DIR + '/{}'.format(my_dataset) + '/{}_comparison_5_epoch_plot_{}_{}_{}_{}.pdf'.format(my_plot_model, my_dataset, 'stochastic', 'quantile', args.history), bbox_inches='tight', dpi=500)

# %%
