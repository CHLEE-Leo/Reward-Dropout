# %%
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import glob
import copy
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from io import BytesIO


parser = argparse.ArgumentParser(description='receive the parameters')
parser.add_argument('--my_seed', type = int, required = True)
parser.add_argument('--dataset', type = str, required = True)       # dataset = {   'sentiment-0', 'sentiment-1', 
                                                                    #               'politeness-0', 'politeness-1',
                                                                    #               'topic-0', 'act-1', 'act-2', 'act-3'}
parser.add_argument('--plot_model', type = str, required = False)   # plot_model : {'all', 'gpt2_small', 'opt', 'xglm', 'gpt2_large', 'gpt2_small_init_weight=uniform'}
parser.add_argument('--reward', type = str, required=True)          # reward : {'r', 'r+log_b', 'r+log_b+log_p'}
parser.add_argument('--dropout', type = str, required=True)         # reward : {'quantile', 'pareto'}
parser.add_argument('--metric', type = str, required = False)       # metric : {'reward', 'acc', 'loss'}
args = parser.parse_args()
my_model = args.plot_model
my_dataset = args.dataset
my_reward = args.reward
my_dropout = args.dropout
my_metric = args.metric + '_history'

# # 모든 model의 결과를 하나의 플롯에 뽑는다면
# if my_plot_model == 'all':
#     model_list = ['gpt2_small', 'opt', 'xglm', 'gpt2_large']

# # 각 model의 결과를 각 플롯에 뽑는다면
# else:
#     model_list = [my_plot_model]
#     model_size = my_model_size

'''
경로 설정
'''
parent_dir = str(Path(os.getcwd()).parents[0])
RESULT_DIR = parent_dir + '/results'
FILE_DIR_LIST = []

'''
파일 불러오기
'''
FILE_DIR_LIST = glob.glob(RESULT_DIR + "/{}/{}/*{}*{}*{}".format(my_dataset, my_model, 'stochastic', my_dropout, my_reward))

# - Filter out the directory with 'quantile_0.85'
FILE_DIR_LIST = [dir for dir in FILE_DIR_LIST if 'quantile_0.85' not in dir]

'''
데이터 로드
'''
# - FILE_DIR_LIST 내 모든 파일 경로를 루프하기
for idx, each_dir in enumerate(FILE_DIR_LIST):

    # - 드롭아웃 사분위수 (e.g., "0.95", "0.9", "0.85", "0.8") 정의하여 my_dropout_rate 이라는 변수로 선언
    each_dir_name_split = np.array(each_dir.split('_'))
    dropout_file_name_idx = np.where(each_dir_name_split == my_dropout)[0][0]
    dropout_rate_file_name_idx = dropout_file_name_idx + 1
    my_dropout_rate = each_dir_name_split[dropout_rate_file_name_idx]
    
    # - my_metric (e.g., "acc_metric", "reward_metric") 파일 로드하여 train_metric_pd 라는 이름의 pandas 변수 선언
    train_metric_pd = pd.read_csv(each_dir + '/' + my_metric + '.csv', index_col=0)

    # - train_metric_pd['model'] = my_model

    # - train_metric_pd 에 dropout_rate 이라는 컬럼 추가 후 my_dropout_rate 값 입력
    train_metric_pd['dropout_rate'] = my_dropout_rate

    # - train_metric_pd 를 행 방향으로 누적 concat() 시키기
    if idx == 0:
        train_metric_pd_all = copy.deepcopy(train_metric_pd)

    else:
        train_metric_pd_all = pd.concat([train_metric_pd_all, train_metric_pd], axis=0)

# - 총 dropout_rate 리스트 및 각 train_metric_pd 의 행 길이 (= epoch 길이) 변수선언
dropout_rate_list = np.unique(train_metric_pd_all['dropout_rate'])
epoch_len = train_metric_pd.shape[0]

# - train_metric_pd_all 의 첫번째 열 (= "train_" + {acc, reward} 열) 에서 최대값과 최소값 구하기
min_val = np.min(train_metric_pd_all.iloc[:, 0])
max_val = np.max(train_metric_pd_all.iloc[:, 0])

'''
전체 범례 설정
'''
# Define labels and colors for the legend (already defined)
# legend_labels = ["q=0.0", "q=0.8", "q=0.9", "q=0.95"]
# legend_colors = ['blue', 'orange', 'green', 'red']

# # Create custom handles for the legend (already defined)
# custom_handles = [Line2D([0], [0], color=legend_colors[i], marker='o', linestyle='None',
#                          markersize=10, label=legend_labels[i]) for i in range(len(legend_labels))]

plt.figure(figsize = (3.5, 2.5))
for i, my_dropout_rate in enumerate(dropout_rate_list):

    # - dropout_rate 별 행 필터링
    train_metric_pd_by_dropout_rate = train_metric_pd_all[train_metric_pd_all['dropout_rate'] == my_dropout_rate]

    # - train_metric_pd_by_dropout_rate 에서 "train_" + {acc, reward} 라는 이름의 열만 분리해 target_metric 라는 변수선언
    target_metric = train_metric_pd_by_dropout_rate['train_{}'.format(my_metric.replace('_history', ''))]

    # - target_metric 플로팅
    plt.plot(target_metric, marker='.', markersize=10, label='{}_{}'.format('quantile', my_dropout_rate))

    # - 만약 my_metric == 'reward_metric' 라면 target_metric 내 최대 값에 해당하는 y축 위치에 평행선 그리기
    if my_model == 'gpt2_small_init_weight=uniform' and my_metric == 'reward_metric':
        cmap = plt.get_cmap("tab10")
        max_val = np.max(target_metric)
        plt.axhline(y=max_val, linestyle='solid', color=cmap(i))

plt.tight_layout()
plt.ylim([min_val, max_val + 0.05])

plt.grid(True)
plt.xticks(fontsize=8, ticks=np.arange(0, epoch_len, 1), labels=np.arange(1, epoch_len+1, 1))
plt.xlabel('epoch')
plt.yticks(fontsize=10, ticks=np.round(np.arange(min_val-0.05, max_val+0.05, 0.05), 1), labels=np.round(np.arange(min_val-0.05, max_val+0.05, 0.05), 1))
# plt.ylabel('mean {}'.format(my_metric.replace('_history', '')))
# plt.ylabel('mean {}'.format('reward'))

plt.legend(bbox_to_anchor = (1.0, 1.0), loc='upper left')
plt.title('{}'.format(my_metric.replace('_history', '')))
if my_metric.replace('_history', '') == 'reward':
    plt.title('$R_{\phi}(\hat{\\tau}), \ \hat{\\tau} \sim \pi_{\\theta}$')
elif my_metric.replace('_history', '') == 'acc':
    plt.title('$\\beta_{\\bar{\\theta}}(\hat{\\tau}), \ \hat{\\tau} \sim \pi_{\\theta}$')

plt.show()
plt.savefig(RESULT_DIR + '/{}'.format(my_dataset) + '/{}'.format(my_model) + '/{}_comparison_plot_{}_{}_{}.pdf'.format(my_metric, 'stochastic', my_dropout, my_reward), bbox_inches='tight')