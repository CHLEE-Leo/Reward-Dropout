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
parser.add_argument('--plot_model', type = str, required = True)   # plot_model : {'gpt2_small', 'gpt2_small_init_weight=uniform'}
parser.add_argument('--dataset', type = str, required = True)   # dataset : {'sentiment', 'topic', 'sentiment+topic'}
parser.add_argument('--reward_type', type = str, required = True)   # reward_type : {'r_', 'r+log_b'}
args = parser.parse_args()

'''
데이터 셋 선택
'''
my_model = args.plot_model
my_dataset = args.dataset
my_reward_type = args.reward_type
# my_dataset = 'sentiment+topic'
# my_reward_type = 'r+log_b'
# my_model = 'gpt2_small'

'''
경로 설정 및 pdf 파일 불러오기
'''
parent_dir = str(Path(os.getcwd()).parents[0])
RESULT_DIR = parent_dir + '/results'
FILE_DIR_LIST = []
# FILE_DIR_LIST += glob.glob(RESULT_DIR + "/{}/{}/*{}*".format(my_dataset, my_model, 'agg'))
for subdir, dirs, files in os.walk(RESULT_DIR): # "agg"를 포함하는 PDF 파일을 모두 찾기
    for file in glob.glob(subdir + '/{}/*{}*agg*.pdf'.format(my_model, my_reward_type)):
        FILE_DIR_LIST += [file]

if my_dataset == 'topic':
    # 토픽 데이터 셋 선택 및 번호에 따라 정렬
    FILE_DIR_LIST = [dir for dir in FILE_DIR_LIST if my_dataset in dir]
    FILE_DIR_LIST = sorted(FILE_DIR_LIST, key=lambda x: int(x.split('/' + my_dataset + '-')[1].split('/')[0]))

elif my_dataset == 'sentiment':
    # 감성 데이터 셋 선택 및 번호에 따라 정렬
    FILE_DIR_LIST = [dir for dir in FILE_DIR_LIST if my_dataset in dir]
    FILE_DIR_LIST = sorted(FILE_DIR_LIST, key=lambda x: (x.split(my_dataset + '-')[1].split("/")[0], x.split("/")[-1]))

elif my_dataset == 'sentiment+topic':
    # 감성 데이터 셋 선택 및 번호에 따라 정렬
    FILE_DIR_LIST = [dir for dir in FILE_DIR_LIST 
                     if "sentiment-0" in dir 
                     or "sentiment-1" in dir 
                     or "topic-0" in dir
                     or "topic-1" in dir 
                     or "topic-2" in dir
                     or "topic-3" in dir]
    FILE_DIR_LIST = sorted(FILE_DIR_LIST, key=lambda x: (x.split('/results/')[1].split('/')[0], x.split('/')[-1].split('_')[0]))

'''
전체 범례 설정
- Define labels and colors for the legend
- Create custom handles for the legend
'''
# legend_labels = ["R($\\tau$)+ln$\\beta(\\tau)\geq0.8$", "R($\\tau$)+ln$\\beta(\\tau)\geq0.9$", "R($\\tau$)+ln$\\beta(\\tau)\geq0.95$", "R($\\tau$)+ln$\\beta(\\tau)$-ln$\\pi(\\tau)>0$", "R($\\tau$)+ln$\\beta(\\tau)$-ln$\\pi(\\tau)$", "(No reward dropout)"]
# legend_colors = ['C0', 'C1', 'C2', 'C3', 'C7', 'C7']
# legend_markers = ['.', '.', '.', '*', 'x', '']

# '''
# 플롯팅 - topic
# '''
# # 토픽의 경우
# custom_handles = [Line2D([0], [0], color=legend_colors[i], marker=legend_markers[i], linestyle='None',
#                          markersize=10, label=legend_labels[i]) for i in range(len(legend_labels))]

# # Create a figure with 1 row and 4 columns for subplots (already defined)
# fig, axs = plt.subplots(2, 4, figsize=(12, 5))      # topic

# # Adjust the spacing between the subplots (already defined)
# plt.subplots_adjust(wspace=0.05, hspace=0)

# # Process each PDF file
# for i, pdf_file in enumerate(FILE_DIR_LIST):

#     if i == 7:
#         j, k = 1, 3  # 수정된 부분
#     else:
#         j, k = i % 2, i // 2

#     # Open the PDF
#     doc = fitz.open(pdf_file)

#     # Extract the first page (or the page you want)
#     page = doc.load_page(0)

#     # Get the image of the page at a higher resolution
#     pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))  # Extract at higher resolution

#     img = BytesIO(pix.tobytes("png"))

#     # Load the high-resolution image into matplotlib
#     img_plot = mpimg.imread(img, format='png')

#     # Plot the image in the respective subplot
#     axs[j][k].imshow(img_plot)
#     axs[j][k].axis('off')

#     # Close the PDF file
#     doc.close()

# # Add a single legend for the whole figure (already defined)
# fig.legend(handles=custom_handles, loc='lower center', 
#         #    ncol=len(custom_handles), bbox_to_anchor=(0.5, -0.03), 
#            ncol=6, bbox_to_anchor=(0.5, -0.02), 
#            fontsize=9.5)

# # Adjust layout (already defined)
# plt.tight_layout(rect=[0, 0.01, 1, 1])

# '''
# 전체 그림 저장
# '''
# SAVE_DIR = RESULT_DIR + '/acc_vs_reward_total_plot.pdf'
# plt.savefig(SAVE_DIR, dpi=500, bbox_inches='tight')


# '''
# 플롯팅 - sentiment
# '''
# # 감성의 경우
# custom_handles = [Line2D([0], [0], color=legend_colors[i], marker=legend_markers[i], linestyle='None',
#                          markersize=8, label=legend_labels[i]) for i in range(len(legend_labels))]

# # Create a figure with 1 row and 4 columns for subplots (already defined)
# fig, axs = plt.subplots(2, 2, figsize=(8, 6))      # sentiment

# # Adjust the spacing between the subplots (already defined)
# plt.subplots_adjust(wspace=0.05, hspace=0)

# # Process each PDF file
# for i, pdf_file in enumerate(FILE_DIR_LIST):

#     j, k = i % 2, i // 2

#     # Open the PDF
#     doc = fitz.open(pdf_file)

#     # Extract the first page (or the page you want)
#     page = doc.load_page(0)

#     # Get the image of the page at a higher resolution
#     pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))  # Extract at higher resolution

#     img = BytesIO(pix.tobytes("png"))

#     # Load the high-resolution image into matplotlib
#     img_plot = mpimg.imread(img, format='png')

#     # Plot the image in the respective subplot
#     axs[j][k].imshow(img_plot)
#     axs[j][k].axis('off')

#     # Close the PDF file
#     doc.close()

# # Add a single legend for the whole figure (already defined)
# fig.legend(handles=custom_handles, loc='upper right', 
#         #    ncol=len(custom_handles), bbox_to_anchor=(0.5, -0.03), 
#            bbox_to_anchor=(1.35, 0.75), 
#            fontsize=14.)

# # Adjust layout (already defined)
# plt.tight_layout(rect=[0, 0.01, 1, 1])

# '''
# 전체 그림 저장
# '''
# SAVE_DIR = RESULT_DIR + '/acc_vs_reward_total_plot_{}.pdf'.format(my_dataset)
# plt.savefig(SAVE_DIR, dpi=500, bbox_inches='tight')

'''
전체 범례 설정
- Define labels and colors for the legend
- Create custom handles for the legend
'''
# legend_labels = ["R($\\tau$)+ln$\\beta(\\tau)\geq$q=0.8", "R($\\tau$)+ln$\\beta(\\tau)\geq$q=0.9", "R($\\tau$)+ln$\\beta(\\tau)\geq$q=0.95", "R($\\tau$)+ln$\\beta(\\tau)$-ln$\\pi(\\tau)>0$", "R($\\tau$)+ln$\\beta(\\tau)$-ln$\\pi(\\tau)$    (No reward dropout)"]
# legend_colors = ['C0', 'C1', 'C2', 'C3', 'C7']
# legend_markers = ['.', '.', '.', '*', 'x']

if my_reward_type == 'r+log_b':
    legend_labels = ["R($\\tau$)+ln$\\beta(\\tau)>0.8$q", "R($\\tau$)+ln$\\beta(\\tau)>0.9$q", "R($\\tau$)+ln$\\beta(\\tau)>0.95$q", "R($\\tau$)+ln$\\beta(\\tau)$-ln$\\pi(\\tau)\geq0.0$q   (without reward dropout)"]
elif my_reward_type == 'r_':
    legend_labels = ["R($\\tau$)$>0.8$q", "R($\\tau$)$>0.9$q", "R($\\tau$)$>0.95$q", "R($\\tau$)+ln$\\beta(\\tau)$-ln$\\pi(\\tau)\geq0.0$q   (without reward dropout)"]
legend_colors = ['C0', 'C1', 'C2', 'C3']
legend_markers = ['.', '.', '.', 'x']

'''
플롯팅 - sentiment+topic
'''
# 토픽의 경우
custom_handles = [Line2D([0], [0], color=legend_colors[i], marker=legend_markers[i], linestyle='None',
                         markersize=10, label=legend_labels[i]) for i in range(len(legend_labels))]

# Create a figure with 1 row and 4 columns for subplots (already defined)
# fig, axs = plt.subplots(2, 4, figsize=(12, 5))      # topic
fig, axs = plt.subplots(2, 6, figsize=(20, 6))      # topic

# Adjust the spacing between the subplots (already defined)
plt.subplots_adjust(wspace=0.05, hspace=0)

# Process each PDF file
for i, pdf_file in enumerate(FILE_DIR_LIST):

    if i == 7:
        j, k = 1, 3  # 수정된 부분
    else:
        j, k = i % 2, i // 2

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
    axs[j][k].imshow(img_plot)
    axs[j][k].axis('off')

    # Close the PDF file
    doc.close()

# Add a single legend for the whole figure (already defined)
fig.legend(handles=custom_handles, loc='lower center', 
        #    ncol=len(custom_handles), bbox_to_anchor=(0.5, -0.03), 
           ncol=6, bbox_to_anchor=(0.5, -0.02), 
           fontsize=18)

# Adjust layout (already defined)
plt.tight_layout(rect=[0, 0.01, 1, 1])

'''
전체 그림 저장
'''
SAVE_DIR = RESULT_DIR + '/acc_vs_reward_total_plot_{}_{}_{}.pdf'.format(my_model, my_dataset, my_reward_type)
plt.savefig(SAVE_DIR, dpi=500, bbox_inches='tight')