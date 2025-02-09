# %%
'''
찐찐 그래프
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# beta와 z에 대한 그리드 생성, 0과 1 사이
beta = np.linspace(0.01, 1, 50)
pi = np.linspace(0.01, 1, 50)

# beta, pi 그리드에 대해 meshgrid 생성
beta, pi = np.meshgrid(beta, pi)

# 주어진 조건에 따라 x 계산
reward = np.log(pi) - np.log(beta)

# 색상 맵 생성
colors = plt.cm.viridis(np.linspace(0, 1, reward.shape[0]))

# 1x2 서브플롯 생성
fig = plt.figure(figsize=(12, 6))

# 3D 플롯 - 크기와 위치 조정
ax1_position = [0.05, 0.15, 0.5, 0.6]  # 3D 플롯의 위치와 크기를 정의
ax1 = fig.add_axes(ax1_position, projection='3d')
ax1.plot_surface(reward, beta, pi, rstride=1, cstride=1, alpha=0.3)
use_color_list = []
for i in range(0, reward.shape[0], 10):
    use_color_list.append(colors[i])
    ax1.plot(reward[i], beta[i], pi[i], color=colors[i], lw=1.5)
    ax1.plot(reward[i], beta[i], zs=0, zdir='z', color=colors[i], lw=1.5, linestyle='--')
ax1.set_xlabel('$R(\\tau)$')
ax1.set_ylabel('$\\beta(\\tau)$')
ax1.set_zlabel('$\\pi(\\tau)$')
ax1.view_init(elev=20., azim=300)  # 3D 플롯의 시점 조정
# ax1.view_init(elev=0., azim=360)  # 3D 플롯의 시점 조정
# ax1.view_init(elev=0., azim=270)  # 3D 플롯의 시점 조정

# 3D 플롯 내에 수식 표시
ax1.text(x=-3, y=0.3, z=1.0, s="$R(\\tau) + \ln\\beta(\\tau) - \ln\\pi(\\tau) = 0$", color='red')

# 2D 플롯 - 크기와 위치 조정
ax2_position = [0.575, 0.20, 0.3 * 0.75, 0.6 * 0.75]  # 2D 플롯의 위치와 크기를 정의
ax2 = fig.add_axes(ax2_position)
ax2.grid(True)  # 그리드 추가

# 정의된 5개의 좌표 쌍
points = [(-2.0, 0.075), (0.05, 0.2), (0.34, 0.3), (0.46, 0.4), (0.5, 0.5)]
labels = ['A', 'B', 'C', 'D', 'E']  # 좌표에 출력할 라벨

# # 각 선과 점을 그리는 부분
# for i, (color, label) in enumerate(zip(use_color_list, labels)):
#     ax2.plot(reward[i*10], beta[i*10], color=color, lw=1.5)  # 선 그리기
#     x, y = points[i]  # 좌표 쌍 선택
#     ax2.plot(x, y, 'o', color=color)  # 점 그리기
#     ax2.text(x, y, f'{label}=({x:.2f}, {y:.2f})', color=color, fontsize=11)  # 좌표 텍스트
# 각 선과 점을 그리는 부분
for i, (color, label) in enumerate(zip(use_color_list, labels)):
    ax2.plot(reward[i*10], beta[i*10], color=color, lw=1.5, label=label)  # 선 그리기
    x, y = points[i]  # 좌표 쌍 선택
    ax2.plot(x, y, 'o', color=color)  # 점 그리기
    ax2.text(x, y, f'{label}=({x:.2f}, {y:.2f})', color=color, fontsize=11)  # 좌표 텍스트

ax2.set_xlabel('$R(\\tau)$')
ax2.set_ylabel('$\\beta(\\tau)$')
ax2.set_xlim(-4.5, 4.5)  # 2D 플롯의 X 축 범위 조정
plt.show()

# %%
'''
# Subplot들을 개별적인 PDF 파일로 저장
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D 플롯 생성
fig1 = plt.figure(figsize=(4, 3))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(reward, beta, pi, rstride=1, cstride=1, alpha=0.3)
for i in range(0, reward.shape[0], 10):
    ax1.plot(reward[i], beta[i], pi[i], color=colors[i], lw=1.5)
    ax1.plot(reward[i], beta[i], zs=0, zdir='z', color=colors[i], lw=1.5, linestyle='--')
ax1.set_xlabel('$R(\\tau)$')
ax1.set_ylabel('$\\beta(\\tau)$')
zlabel = ax1.set_zlabel('$\\pi(\\tau)$')
ax1.view_init(elev=20, azim=300)
ax1.text(x=-5.25, y=0.2, z=0.9, s="$R(\\tau) + \ln\\beta(\\tau) - \ln\\pi(\\tau) = 0$", color='red')

# PDF 파일로 저장
fig1.savefig('Equilibrium_Plane_3D.pdf', bbox_inches='tight', pad_inches=0.1, bbox_extra_artists=[zlabel])

# %%
'''
1) 2D 플레인 첫번째 그림
'''
# 색상 맵 생성
colors = plt.cm.viridis(np.linspace(0, 1, reward.shape[0]))
use_color_list = []
for i in range(0, reward.shape[0], 10):
    use_color_list.append(colors[i])

# 정의된 5개의 좌표 쌍
points = [(-2.0, 0.075), (0.05, 0.2), (0.34, 0.3), (0.46, 0.4), (0.5, 0.5)]
labels = ['A', 'B', 'C', 'D', 'E']  # 좌표에 출력할 라벨

# 2D 플롯 저장
fig2 = plt.figure(figsize=(4, 3))
ax2_copy = fig2.add_subplot(111)
ax2_copy.grid(True)
for i, (color, label) in enumerate(zip(use_color_list, labels)):
    ax2_copy.plot(reward[i*10], beta[i*10], color=color, lw=1.5)
    x, y = points[i]
    ax2_copy.plot(x, y, 'o', color=color)
    ax2_copy.text(x+0.02, y+0.02, f'{label}=({x:.2f}, {y:.2f})', color=color, fontsize=11)
ax2_copy.set_xlabel('$R(\\tau)$', fontsize=14)
ax2_copy.set_ylabel('$\\beta(\\tau)$', fontsize=14)
ax2_copy.set_xlim(-4.5, 4.5)

# 여백 조정
fig2.tight_layout() # 레이아웃 맞춤
# ax2_copy.legend(fontsize='large')
fig2.savefig('Equilibrium_Plane_2D-1.pdf')

# %%
'''
2) 2D 플레인 두번째 그림
'''
# 색상 맵 생성
colors = plt.cm.viridis(np.linspace(0, 1, reward.shape[0]))
use_color_list = []
for i in range(0, reward.shape[0], 10):
    use_color_list.append(colors[i])

# 정의된 5개의 좌표 쌍
points = [(-3.0, 0.2), (0.0, 0.2), (0.8, 0.2), (1.125, 0.2), (1.4, 0.2)]
labels = ['A', 'B', 'C', 'D', 'E']  # 좌표에 출력할 라벨

# 2D 플롯 저장
fig2 = plt.figure(figsize=(4, 3))
ax2_copy = fig2.add_subplot(111)
ax2_copy.grid(True)
for i, (color, label) in enumerate(zip(use_color_list, labels)):
    ax2_copy.plot(reward[i * 10], beta[i * 10], color=color, lw=1.5)  # 선 그리기
    x, y = points[i]  # 좌표 쌍 선택
    ax2_copy.plot(x, y, 'o', color=color)  # 점 그리기
    # ax2_copy.text(x + 0.02, y + 0.02, f'{label}=({x:.2f}, {y:.2f})', color=color, fontsize=11)  # 좌표 텍스트
    ax2_copy.plot([], [], 'o', color=color, label=f'{label} = ({x:.2f}, {y:.2f})')  # 범례 항목 추가

ax2_copy.set_xlabel('$R(\\tau)$', fontsize=14)
ax2_copy.set_ylabel('$\\beta(\\tau)$', fontsize=14)
ax2_copy.set_xlim(-4.5, 4.5)
# ax2_copy.hlines(0.2, -3.0, 1.4, color='red', linestyle=':', linewidth=1)
# ax2_copy.hlines(0.2, -4.0, 4.0, color='red', linestyle='--', linewidth=1)

# 여백 조정, 범례 지정 및 플롯 저장
fig2.tight_layout()
ax2_copy.legend(fontsize='small')
fig2.savefig('Equilibrium_Plane_2D-2.pdf')
plt.show()

# %%
'''
3) 2D 플레인 세번째 그림
'''
# 색상 맵 생성
colors = plt.cm.viridis(np.linspace(0, 1, reward.shape[0]))
use_color_list = []
for i in range(0, reward.shape[0], 10):
    use_color_list.append(colors[i])

# 정의된 5개의 좌표 쌍
points = [(0, 0.02), (0, 0.2), (0, 0.4), (0, 0.6), (0, 0.8)]
labels = ['A', 'B', 'C', 'D', 'E']  # 좌표에 출력할 라벨

# 2D 플롯 저장
fig2 = plt.figure(figsize=(4, 3))
ax2_copy = fig2.add_subplot(111)
ax2_copy.grid(True)
for i, (color, label) in enumerate(zip(use_color_list, labels)):
    ax2_copy.plot(reward[i * 10], beta[i * 10], color=color, lw=1.5)  # 선 그리기
    x, y = points[i]  # 좌표 쌍 선택
    ax2_copy.plot(x, y, 'o', color=color)  # 점 그리기
    # ax2_copy.text(x + 0.02, y + 0.02, f'{label}=({x:.2f}, {y:.2f})', color=color, fontsize=11)  # 좌표 텍스트
    ax2_copy.plot([], [], 'o', color=color, label=f'{label} = ({x:.2f}, {y:.2f})')  # 범례 항목 추가

ax2_copy.set_xlabel('$R(\\tau)$', fontsize=14)
ax2_copy.set_ylabel('$\\beta(\\tau)$', fontsize=14)
ax2_copy.set_xlim(-4.5, 4.5)

# 여백 조정, 범례 지정 및 플롯 저장
fig2.tight_layout()
ax2_copy.legend(fontsize='small')
fig2.savefig('Equilibrium_Plane_2D-3.pdf')
plt.show()

# %%
'''
4) 2D 플레인 네번째 그림
'''
# 색상 맵 생성
colors = plt.cm.viridis(np.linspace(0, 1, reward.shape[0]))
use_color_list = []
for i in range(0, reward.shape[0], 10):
    use_color_list.append(colors[i])

# 정의된 5개의 좌표 쌍 (우하향 배열, 선형 패턴)
points = [(-3.95, 0.52), (-0.27, 0.27), (0.7, 0.2), (1.4, 0.15), (2.0, 0.12)]
labels = ['A', 'B', 'C', 'D', 'E']  # 좌표에 출력할 라벨

# 2D 플롯 저장
fig2 = plt.figure(figsize=(4, 3))
ax2_copy = fig2.add_subplot(111)
ax2_copy.grid(True)
for i, (color, label) in enumerate(zip(use_color_list, labels)):
    ax2_copy.plot(reward[i * 10], beta[i * 10], color=color, lw=1.5)  # 선 그리기
    x, y = points[i]  # 좌표 쌍 선택
    ax2_copy.plot(x, y, 'o', color=color)  # 점 그리기
    # ax2_copy.text(x + 0.02, y + 0.02, f'{label}=({x:.2f}, {y:.2f})', color=color, fontsize=11)  # 좌표 텍스트
    ax2_copy.plot([], [], 'o', color=color, label=f'{label} = ({x:.2f}, {y:.2f})')  # 범례 항목 추가

ax2_copy.set_xlabel('$R(\\tau)$', fontsize=14)
ax2_copy.set_ylabel('$\\beta(\\tau)$', fontsize=14)
ax2_copy.set_xlim(-4.5, 4.5)

# 여백 조정, 범례 지정 및 플롯 저장
fig2.tight_layout()
ax2_copy.legend(fontsize='small')
fig2.savefig('Equilibrium_Plane_2D-4.pdf')
plt.show()

# %%
'''
5) 2D 플레인 다섯번째 그림
'''
# 색상 맵 생성
colors = plt.cm.viridis(np.linspace(0, 1, reward.shape[0]))
use_color_list = []
for i in range(0, reward.shape[0], 10):
    use_color_list.append(colors[i])

# 정의된 5개의 좌표 쌍 (우상향 배열, 선형 패턴)
points = [(-2, 0.08), (-0.54, 0.36), (-0.10, 0.45), (0.19, 0.51), (0.4, 0.55)]
labels = ['A', 'B', 'C', 'D', 'E']  # 좌표에 출력할 라벨

# 2D 플롯 저장
fig2 = plt.figure(figsize=(4, 3))
ax2_copy = fig2.add_subplot(111)
ax2_copy.grid(True)

for i, (color, label) in enumerate(zip(use_color_list, labels)):
    ax2_copy.plot(reward[i * 10], beta[i * 10], color=color, lw=1.5)  # 선 그리기
    x, y = points[i]  # 좌표 쌍 선택
    ax2_copy.plot(x, y, 'o', color=color)  # 점 그리기
    ax2_copy.plot([], [], 'o', color=color, label=f'{label} = ({x:.2f}, {y:.2f})')  # 범례 항목 추가

ax2_copy.set_xlabel('$R(\\tau)$', fontsize=14)
ax2_copy.set_ylabel('$\\beta(\\tau)$', fontsize=14)
ax2_copy.set_xlim(-4.5, 4.5)

# 여백 조정, 범례 지정 및 플롯 저장
fig2.tight_layout()
ax2_copy.legend(fontsize='small')
fig2.savefig('Equilibrium_Plane_2D-5.pdf')
plt.show()


# %%
'''
6) 2D 플레인 여섯번째 그림
'''
# 색상 맵 생성
colors = plt.cm.viridis(np.linspace(0, 1, reward.shape[0]))
use_color_list = []
for i in range(0, reward.shape[0], 10):
    use_color_list.append(colors[i])

# 정의된 5개의 좌표 쌍
points = [(-1.35, 0.8), (-1.0, 0.57), (-0.54, 0.36), (0.19, 0.175), (1.5, 0.05)]
labels = [r'$\mathrm{B}_{1}$', r'$\mathrm{B}_{2}$', r'$\mathrm{B}_{3}$', r'$\mathrm{B}_{4}$', r'$\mathrm{B}_{5}$']  # 좌표에 출력할 라벨

# 색상 맵 생성
colors = plt.cm.viridis(np.linspace(0, 1, len(points)))
use_color_list = ['lightgrey'] * len(points)
use_color_list[1] = colors[1]  # 두 번째 선의 색을 유지
use_color_list_points = [colors[1]] *  len(points)

# 2D 플롯 저장
fig2 = plt.figure(figsize=(4.5, 3.325))
ax2_copy = fig2.add_subplot(111)
ax2_copy.grid(True)

for i, (color, label) in enumerate(zip(use_color_list, labels)):
    ax2_copy.plot(reward[i * 10], beta[i * 10], color=color, lw=1.5)  # 선 그리기
    x, y = points[i]  # 좌표 쌍 선택
    ax2_copy.plot(x, y, 'o', color=use_color_list_points[i])  # 점 그리기
    ax2_copy.text(x+0.02, y+0.02, f'{label}=({x:.2f}, {y:.2f})', color=use_color_list_points[i], fontsize=12)

ax2_copy.set_xlabel('$R(\\tau)$', fontsize=14)
ax2_copy.set_ylabel('$\\beta(\\tau)$', fontsize=14)
ax2_copy.set_xlim(-4.5, 4.5)

# 여백 조정, 범례 지정 및 플롯 저장
fig2.tight_layout()
fig2.savefig('Equilibrium_Plane_2D-6.pdf')
# plt.show()
# %%
'''
추가 분석
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import copy
from scipy.interpolate import interp1d

# beta와 pi에 대한 그리드 생성, 0과 1 사이
beta = np.linspace(0.01, 1, 50)
pi = np.linspace(0.01, 1, 50)

# beta, pi 그리드에 대해 meshgrid 생성
beta, pi = np.meshgrid(beta, pi)

# 주어진 조건에 따라 reward 계산
reward = np.log(pi) - np.log(beta)

# 색상 맵 생성
colors = plt.cm.viridis(np.linspace(0, 1, reward.shape[0]))

# 1x3 서브플롯 생성
fig = plt.figure(figsize=(8, 6))

# 3D 플롯 - 크기와 위치 조정
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot_surface(reward, beta, pi, rstride=1, cstride=1, alpha=0.3)

use_color_list = []
for i in range(0, reward.shape[0], 10):
    use_color_list.append(colors[i])
    ax1.plot(reward[i], beta[i], pi[i], color=colors[i], lw=1.5)
    ax1.plot(reward[i], beta[i], zs=0, zdir='z', color=colors[i], lw=1.5, linestyle='--')
ax1.set_xlabel('$R(\\tau)$')
ax1.set_ylabel('$\\beta(\\tau)$')
ax1.set_zlabel('$\\pi(\\tau)$')
ax1.view_init(elev=20, azim=300)

# 3D 플롯 내에 수식 표시
ax1.text(x=-3, y=0.3, z=1.0, s="$R(\\tau) + \ln\\beta(\\tau) - \ln\\pi(\\tau) = 0$", color='red')

plt.tight_layout()

# (0) 기존 2D 플롯 - Reward vs Beta
# - 정의된 5개의 좌표 쌍
points = [(-2.0, 0.075), (0.05, 0.2), (0.34, 0.3), (0.46, 0.4), (0.5, 0.5)]
labels = ['A', 'B', 'C', 'D', 'E']  # 좌표에 출력할 라벨

ax2 = fig.add_subplot(222)
ax2.grid(True)
for i, (color, label) in enumerate(zip(colors[::10], labels)):
    ax2.plot(reward[i*10], beta[i*10], color=color, lw=1.5)
    ax2.plot(points[i][0], points[i][1], 'o', color=color)
    ax2.text(points[i][0], points[i][1], f'{label}=({points[i][0]:.2f}, {points[i][1]:.2f})', color=color, fontsize=11)
ax2.set_xlabel('$R(\\tau)$')
ax2.set_ylabel('$\\beta(\\tau)$')
ax2.set_xlim(-4.5, 4.5)
plt.tight_layout()

# (1) 추가된 2D 플롯 - Pi vs Beta (pi = beta)
# - 정의된 5개의 좌표 쌍
points = [(-2.0, 0.075), (0.05, 0.2), (0.34, 0.3), (0.46, 0.4), (0.5, 0.5)]
points = np.array(points)
reward_coordinates = points[:, 0]
beta_coordinates = points[:, 1]
pi_coordinates = beta_coordinates * np.exp(reward_coordinates)
new_points1 = list(zip(beta_coordinates, pi_coordinates))
labels = ['A', 'B', 'C', 'D', 'E']  # 좌표에 출력할 라벨


ax3 = fig.add_subplot(223)
ax3.grid(True)

ax3.plot(beta[14], pi.T[14], color='black', lw=1.)
ax3.plot(beta, pi, color=cm.viridis(0.3), lw=1., alpha=0.4)
for i in range(0, beta.shape[0], 10):
    # ax3.plot(beta[i], pi.T[i], color=cm.viridis(0.3), lw=1., alpha=0.4)
    ax3.plot(new_points1[i//10][0], new_points1[i//10][1], 'o', color=use_color_list[i//10])
    ax3.text(new_points1[i//10][0]+0.04, new_points1[i//10][1]-0.015, f'{labels[i//10]}=({new_points1[i//10][0]:.2f}, {new_points1[i//10][1]:.2f})', color=use_color_list[i//10], fontsize=11)

ax3.set_xlim(-0.035, 1.)
ax3.set_ylim(-0.035, 1.)
ax3.set_xlabel('$\\beta(\\tau)$')
ax3.set_ylabel('$\\pi(\\tau)$')
plt.tight_layout()

# (2) 추가된 2D 플롯 - Pi vs Reward (pi = beta*exp(reward))
# - 정의된 5개의 좌표 쌍
points = [(-2.0, 0.075), (0.05, 0.2), (0.34, 0.3), (0.46, 0.4), (0.5, 0.5)]
points = np.array(points)
reward_coordinates = points[:, 0]
beta_coordinates = points[:, 1]
pi_coordinates = beta_coordinates * np.exp(reward_coordinates)

new_points2 = list(zip(reward_coordinates, pi_coordinates))
labels = ['A', 'B', 'C', 'D', 'E']  # 좌표에 출력할 라벨

ax4 = fig.add_subplot(224)
ax4.grid(True)

ax4.plot(reward.T[14], pi.T[14], color='black', lw=1.)
ax4.plot(reward, pi, color=cm.viridis(0.3), lw=1., alpha=0.4)
# ax4.plot(reward.T[14], pi.T[14], color='black', lw=1.)
for i in range(0, reward.shape[0], 10):
    ax4.plot(new_points2[i//10][0], new_points2[i//10][1], 'o', color=use_color_list[i//10])
    ax4.text(new_points2[i//10][0]+0.02, new_points2[i//10][1]+0.02, f'{labels[i//10]}=({new_points2[i//10][0]:.2f}, {new_points2[i//10][1]:.2f})', color=use_color_list[i//10], fontsize=11)

ax4.set_xlim(-4.5, 4.5)
ax4.set_ylim(-0.035, 1.)
ax4.set_xlabel('$R(\\tau)$')
ax4.set_ylabel('$\\pi(\\tau)$')

plt.tight_layout()
plt.show()



























# %%
'''
ddd
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# beta와 z에 대한 그리드 생성, 0과 1 사이
beta = np.linspace(0.01, 1, 50)
pi = np.linspace(0.01, 1, 50)

# beta, pi 그리드에 대해 meshgrid 생성
beta, pi = np.meshgrid(beta, pi)

# 주어진 조건에 따라 x 계산
reward = np.log(pi) - np.log(beta)

# 색상 맵 생성
colors = plt.cm.viridis(np.linspace(0, 1, reward.shape[0]))

# 3D 플롯 생성
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(reward, beta, pi, rstride=1, cstride=1, alpha=0.3)
for i in range(0, reward.shape[0], 10):
    ax.plot(reward[i], beta[i], pi[i], color=colors[i], lw=1.5)
    ax.plot(reward[i], beta[i], zs=0, zdir='z', color=colors[i], lw=1.5, linestyle='--')


# pi[0], pi[10], pi[20], pi[30], pi[40]

# 화살표의 시작점과 끝점 좌표
arrows = [
    ((-2.00, 0.07, pi[0][0]), (0.05, 0.20, pi[10][0])),
    ((0.05, 0.20, pi[10][0]), (0.34, 0.30, pi[20][0])),
    ((0.34, 0.30, pi[20][0]), (0.46, 0.40, pi[30][0])),
    ((0.46, 0.40, pi[30][0]), (0.50, 0.50, pi[40][0]))
]

# 각 화살표에 대해 quiver 함수를 사용하여 화살표 추가
for start, end in arrows:
    ax.quiver(start[0], start[1], start[2], end[0]-start[0], end[1]-start[1], end[2]-start[2], arrow_length_ratio=0.2, color='r')

ax.set_xlabel('$R(\\tau)$')
ax.set_ylabel('$\\beta(\\tau)$')
ax.set_zlabel('$\\pi(\\tau)$')
ax.view_init(elev=20, azim=210)
# ax.view_init(elev=20, azim=300)
ax.text(x=-5.25, y=0.2, z=0.9, s="$R(\\tau) + \ln\\beta(\\tau) - \ln\\pi(\\tau) = 0$", color='red')

plt.show()
