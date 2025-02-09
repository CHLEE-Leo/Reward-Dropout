# %%
import os
from pathlib import Path
import glob
import itertools
import pandas as pd

def save_result_table(parent_dir, result_dir, data_dir, 
                      rl_model, ref_model, 
                      decoding, n_epoch, lr, 
                      dropout_list, dropout_rate_list):

    # None과 0.0의 배타적인 조합 추가
    combinations = [('None', 0.0)]
    combinations += [(d, r) for d in dropout_list if d != 'None' for r in dropout_rate_list if r != 0.0]

    folder_dir = os.path.join(parent_dir, result_dir, data_dir, rl_model)
    all_folders = []

    # 모든 조합 생성 및 폴더 탐색
    for dropout, dropout_rate in combinations:
        pattern = '*ref={}*ref_dec={}*n_epoch={}*lr={}*dropout={}*dropout_rate={}'.format(ref_model, decoding, n_epoch, lr, dropout, dropout_rate)
        subfolder_dir = glob.glob(os.path.join(folder_dir, pattern))
        all_folders += subfolder_dir

    # 초기화
    acc_dfs = []
    reward_dfs = []

    # 각 폴더의 CSV 파일 읽기
    for folder in all_folders:
        acc_file = os.path.join(folder, 'acc_history.csv')
        reward_file = os.path.join(folder, 'reward_history.csv')

        if os.path.exists(acc_file):
            acc_df = pd.read_csv(acc_file).iloc[-1, 1:]   # 마지막 에포크 값만 + train_acc 컬럼만
            acc_dfs.append(acc_df)

        if os.path.exists(reward_file):
            reward_df = pd.read_csv(reward_file).iloc[-1, 1:]   # 마지막 에포크 값만 + train_reward 컬럼만
            reward_dfs.append(reward_df)

    # 행 방향으로 CSV 파일 병합
    total_acc_history = pd.concat(acc_dfs, axis=0).reset_index(drop=True)
    total_reward_history = pd.concat(reward_dfs, axis=0).reset_index(drop=True)

    # 열 방향으로 CSV 파일 병합
    final_total_history = pd.concat([total_acc_history, total_reward_history], axis=1)
    final_total_history.columns = ['train_acc', 'train_reward']

    # dropout 및 dropout_rate DF 만들기
    info_df = pd.DataFrame(combinations, columns=['dropout', 'dropout_rate'])

    # final_total_history와 info_df 열방향 병합
    result_table = pd.concat([info_df, final_total_history], axis=1)

    # 결과 출력
    print(result_table)

    # 파일 저장
    file_name = 'result_table.csv'
    result_table.to_csv(os.path.join(folder_dir, file_name))

# %%
if __name__ == '__main__':
    parent_dir = str(Path(os.getcwd()).parents[0])
    result_dir = 'results'
    # data_dir_list = ['sentiment-0', 'sentiment-1', 'topic-0', 'topic-1', 'topic-2', 'topic-3']
    data_dir_list = ['sentiment-0', 'sentiment-1']
    # data_dir_list = ['topic-0', 'topic-1', 'topic-2', 'topic-3']
    rl_model = 'gpt2_small'
    ref_model = 'opt_large'
    decoding = 'stochastic'
    n_epoch = 5
    lr = 5e-06
    dropout_list = ['None', 'random', 'quantile']
    dropout_rate_list = [0.0, 0.2, 0.4, 0.6, 0.8]

    for data_dir in data_dir_list:
        save_result_table(parent_dir,
                        result_dir,
                        data_dir,
                        rl_model,
                        ref_model,
                        decoding,
                        n_epoch,
                        lr,
                        dropout_list,
                        dropout_rate_list)