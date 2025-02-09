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
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from io import BytesIO



parser = argparse.ArgumentParser(description='receive the parameters')
parser.add_argument('--my_seed', type = int, required = True)
parser.add_argument('--dataset', type = str, required = True)       # dataset = {   'sentiment-0', 'sentiment-1', 
                                                                    #               'toxicity-0', 'toxicity-1', 
                                                                    #               'politeness-0', 'politeness-1',
                                                                    #               'emotion-0', 'emotion-1', 'emotion-2', 'emotion-3', 'emotion-4', 'emotion-5', 'emotion-6',
                                                                    #               'topic-0', 'act-1', 'act-2', 'act-3'}
parser.add_argument('--history', type = str, required = False)      # history : {'reward', 'acc', 'loss'}
args = parser.parse_args()
my_dataset = args.dataset
my_history = args.history

'''
경로 설정
'''
parent_dir = str(Path(os.getcwd()).parents[0])
RESULT_DIR = parent_dir + '/results'
# REWARD_FILE_DIR_LIST = []
# ACC_FILE_DIR_LIST = []

'''
pdf 파일 불러오기
'''
pdf_files = glob.glob(RESULT_DIR + '/{}/[!all]*stochastic_quantile_{}.pdf'.format(my_dataset, my_history))
# REWARD_FILE_DIR_LIST = glob.glob(RESULT_DIR + '/{}/[!all]*stochastic_quantile_reward.pdf'.format(my_dataset))
# ACC_FILE_DIR_LIST = glob.glob(RESULT_DIR + '/{}/[!all]*stochastic_quantile_acc.pdf'.format(my_dataset))

# # List of your PDF files
# pdf_files = [
#     '/home/messy92/Leo/NAS_folder/controllability-of-LM/results/topic-1/gpt2_small_comparison_5_epoch_plot_topic-1_stochastic_quantile_reward.pdf',
#     '/home/messy92/Leo/NAS_folder/controllability-of-LM/results/topic-1/opt_comparison_5_epoch_plot_topic-1_stochastic_quantile_reward.pdf',
#     '/home/messy92/Leo/NAS_folder/controllability-of-LM/results/topic-1/xglm_comparison_5_epoch_plot_topic-1_stochastic_quantile_reward.pdf',
#     '/home/messy92/Leo/NAS_folder/controllability-of-LM/results/topic-1/gpt2_large_comparison_5_epoch_plot_topic-1_stochastic_quantile_reward.pdf'
#     ]

'''
전체 범례 설정
'''
# Define labels and colors for the legend (already defined)
# legend_labels = ["q=0.95", "q=0.9", "q=0.8", "q=0.0"]
# legend_colors = ['red', 'green', 'orange', 'blue']
# legend_labels = ["q=0.0", "q=0.8", "q=0.85", "q=0.9", "q=0.95"]
# legend_colors = ['blue', 'orange', 'green', 'red', 'purple']
legend_labels = ["q=0.0", "q=0.8", "q=0.9", "q=0.95"]
legend_colors = ['blue', 'orange', 'green', 'red']


# Create custom handles for the legend (already defined)
custom_handles = [Line2D([0], [0], color=legend_colors[i], marker='o', linestyle='None',
                         markersize=10, label=legend_labels[i]) for i in range(len(legend_labels))]


'''
플롯팅
'''
# Create a figure with 1 row and 4 columns for subplots (already defined)
fig, axs = plt.subplots(1, 4, figsize=(12.5, 5))

# Adjust the spacing between the subplots (already defined)
plt.subplots_adjust(wspace=0.05, hspace=0)

# Process each PDF file
for i, pdf_file in enumerate(pdf_files):
    # Open the PDF
    doc = fitz.open(pdf_file)

    # Extract the first page (or the page you want)
    page = doc.load_page(0)

    # Get the image of the page at a higher resolution
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))  # Extract at higher resolution

    img = BytesIO(pix.tobytes("png"))

    # Load the high-resolution image into matplotlib
    img_plot = mpimg.imread(img, format='png')

    # Plot the image in the respective subplot
    axs[i].imshow(img_plot)
    axs[i].axis('off')

    # Close the PDF file
    doc.close()

# Add a single legend for the whole figure (already defined)
fig.legend(handles=custom_handles, loc='lower center', ncol=len(custom_handles), bbox_to_anchor=(0.5, .1), fontsize=16)

# Adjust layout (already defined)
plt.tight_layout(rect=[0, 0.01, 1, 1])

# Save the composite plot as a high-resolution PDF (already defined)
plt.savefig('/home/messy92/Leo/NAS_folder/controllability-of-LM/results/{}/{}_composite_plot_{}.pdf'.format(my_dataset, my_dataset, my_history), dpi=500, bbox_inches='tight')




# '''
# 플롯팅
# '''
# model_list = ['gpt2_small', 'opt', 'xglm', 'gpt2_large']
# nrows = 2
# ncols = 4
# fig, ax1 = plt.subplots(figsize=(12, 5), nrows=nrows, ncols=ncols, squeeze=False)

# for subplot_idx, my_model in range(model_list)
#     synthetic_data = pd.read_csv(parent_dir + '/prep_data/position-game/train_samples_mra={}_alpha={}_beta={}.csv'.format(args.min_reward_action, reward_alpha, reward_beta), index_col=0)

#     '''
#     gen 데이터 case 별로 분리하기
#     '''    
#     syn_data_case = synthetic_data[synthetic_data['case'] == target_case]
#     bp_actions, bp_counts = np.unique(syn_data_case['bp_sampled_actions'], return_counts = True)

#     '''
#     test 데이터 불러오기 및 case 별로 분리하기
#     '''
#     file_dir = '{}_{}_{}_{}_mra={}_alpha={}_beta={}'.format(batch_size, lr, num_epoch, target_case, args.min_reward_action, reward_alpha, reward_beta)

#     print('file_dir :', file_dir)
#     all_test_action_dist_pd = pd.read_csv(parent_dir + '/results/position-game/TargetPolicy/' + file_dir + '/test_action_dist_mra={}_alpha={}_beta={}.csv'.format(args.min_reward_action, reward_alpha, reward_beta), index_col=0)

#     tp_actions = all_test_action_dist_pd.columns
#     tp_actions = np.array(tp_actions).astype('int32')
#     tp_counts = all_test_action_dist_pd.iloc[0]
#     tp_counts = np.array(tp_counts).astype('int32')

#     '''
#     test의 보상 결과 데이터 불러오기
#     '''
#     print('file_dir :', file_dir)
#     all_test_reward_dist_pd = pd.read_csv(parent_dir + '/results/position-game/TargetPolicy/' + file_dir + '/test_reward_dist_mra={}_alpha={}_beta={}.csv'.format(args.min_reward_action, reward_alpha, reward_beta), index_col=0)
#     mean_test_reward = tf.reduce_mean(all_test_reward_dist_pd)

#     # reward 분포 플로팅
#     reward_barplot = ax1[0][subplot_idx].bar(supports, reward_dist, align='center', width = .7, alpha=0.5, label='reward ($\geq 6$)', color='red')
#     ax1[0][subplot_idx].set_xticks(ticks=supports, labels=supports, fontsize=14)
#     ax1[0][subplot_idx].set_yticks(ticks=np.round(np.arange(0, 0.8, 0.2), 1), labels=np.round(np.arange(0, 0.8, 0.2), 1), fontsize=14)
#     ax1[0][subplot_idx].set_ylabel('reward', rotation=270, labelpad=15, fontsize=14)

#     # behavior-policy 분포 플로팅
#     ax2 = ax1[0][subplot_idx].twinx()
#     bp_action_barplot = ax2.bar(bp_actions, bp_counts, align='center', width = .7, alpha=0.5, label='behavior policy ($\\beta$)', color='dodgerblue')
#     ax2.set_xlabel('action', fontsize=14)
#     ax2.set_ylabel('likelihood', fontsize=14)

#     # target policy 분포 플로팅
#     tp_action_barplot = ax2.bar(tp_actions, tp_counts, align='center', width = .3, label='target policy ($\\theta$)', color='green', alpha = .7, edgecolor='black')
#     for idx, tp_bar in enumerate(tp_action_barplot):
#         tp_action_barplot[idx].set_hatch("/" * 5)
#     ax2.set_yticks(ticks=np.arange(0, sample_size + num_epi, 2 * num_epi), labels=np.round(np.arange(0, 1.2, 0.2), 1), fontsize=14)

#     # 평균 보상
#     ax2.text(0.035, 0.975, "$\\tilde{{r}}={}$".format(np.round(mean_test_reward, 2)), ha='left', va='top', transform=ax2.transAxes, fontsize=14)

#    # 서브플롯 타이틀
#     ax2.set_title('Case {}:\n alpha={}, beta={}'.format(int(subplot_idx), reward_alpha, reward_beta), fontsize=14)


# # # 수퍼 타이틀
# # fig.suptitle('Case {} : $p_\\beta \sim \mathcal{{N}}(\mu={}, \sigma={})$'.format(target_case, 5, 0.5), fontsize=16)

# # 레이아웃 맞춤
# fig.tight_layout(pad=1.)

# # 저장하기
# fig.savefig(parent_dir + '/4col_figure' + file_dir + '.pdf')