import matplotlib.pyplot as plt
import numpy as np

# --- 样式设置：提升学术质感 ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 13,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 200
})

# --- 数据录入 ---
alpha_labels = ['0.1', '0.2', '0.5', '1.0', '2.0']
alpha_vals = [0, 1, 2, 3, 4]
methods = ['Retrain', 'FAIR-VUE', 'PGA', 'CONDA', 'Fast-fU', 'QuickDrop', 'Fed_eraser']

test_acc = {
    'Retrain': [0.6937, 0.7036, 0.7334, 0.7580, 0.7850],
    'FAIR-VUE': [0.6934, 0.6968, 0.7070, 0.7320, 0.7620],
    'PGA': [0.6938, 0.6879, 0.6703, 0.6850, 0.7120],
    'CONDA': [0.5937, 0.6150, 0.6788, 0.7120, 0.7380],
    'Fast-fU': [0.6325, 0.6194, 0.5802, 0.5650, 0.5520],
    'QuickDrop': [0.6449, 0.6535, 0.6792, 0.6910, 0.7150],
    'Fed_eraser': [0.6646, 0.6637, 0.6610, 0.6580, 0.6750]
}

target_acc = {
    'Retrain': [0.6691, 0.6813, 0.7179, 0.7410, 0.7650],
    'FAIR-VUE': [0.7074, 0.7094, 0.7155, 0.7380, 0.7610],
    'PGA': [0.7063, 0.7247, 0.7800, 0.7920, 0.8150],
    'CONDA': [0.7781, 0.7971, 0.8540, 0.8750, 0.8950],
    'Fast-fU': [0.7652, 0.7619, 0.7521, 0.7310, 0.7250],
    'QuickDrop': [0.7855, 0.7956, 0.8261, 0.8420, 0.8650],
    'Fed_eraser': [0.7508, 0.7493, 0.7447, 0.7380, 0.7520]
}

mia_f1 = {
    'Retrain': [0.415, 0.444, 0.531, 0.585, 0.612],
    'FAIR-VUE': [0.436, 0.441, 0.456, 0.482, 0.515],
    'PGA': [0.329, 0.310, 0.254, 0.225, 0.195],
    'CONDA': [0.512, 0.469, 0.341, 0.285, 0.245],
    'Fast-fU': [0.505, 0.503, 0.496, 0.475, 0.455],
    'QuickDrop': [0.527, 0.507, 0.447, 0.405, 0.375],
    'Fed_eraser': [0.424, 0.409, 0.366, 0.325, 0.295]
}

time_costs = {
    'Retrain': [3792, 3676, 3328, 3120, 2950],
    'FAIR-VUE': [3.89, 4.11, 4.76, 5.25, 6.15],
    'PGA': [32.78, 32.75, 32.65, 32.50, 32.40],
    'CONDA': [42.44, 41.80, 39.88, 38.50, 37.20],
    'Fast-fU': [29.74, 27.32, 20.05, 18.20, 16.50],
    'QuickDrop': [4.28, 4.62, 5.64, 6.15, 6.95],
    'Fed_eraser': [22.80, 22.91, 23.24, 23.50, 23.90]
}

# --- 绘图配置 ---
fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

# 颜色和样式配置
style_config = {
    'Retrain': {'color': '#333333', 'marker': '', 'ls': '--', 'lw': 2, 'alpha': 0.8},
    'FAIR-VUE': {'color': '#E31A1C', 'marker': '*', 'ls': '-', 'lw': 3, 'alpha': 1.0},
    'PGA': {'color': '#1F78B4', 'marker': 'o', 'ls': '-', 'lw': 1.5, 'alpha': 0.7},
    'CONDA': {'color': '#33A02C', 'marker': '^', 'ls': '-', 'lw': 1.5, 'alpha': 0.7},
    'Fast-fU': {'color': '#FF7F00', 'marker': 'v', 'ls': '-', 'lw': 1.5, 'alpha': 0.7},
    'QuickDrop': {'color': '#6A3D9A', 'marker': 'D', 'ls': '-', 'lw': 1.5, 'alpha': 0.7},
    'Fed_eraser': {'color': '#B15928', 'marker': 'p', 'ls': '-', 'lw': 1.5, 'alpha': 0.7},
}

metrics = [test_acc, target_acc, mia_f1, time_costs]
# 修改此处：在 Target Accuracy 后加上 ↓
titles = ['(a) Test Accuracy ↑', '(b) Target Accuracy ↓', '(c) MIA F1 Score ↓', '(d) Time (s) ↓']
ylabels = ['Accuracy', 'Accuracy', 'F1 Score', 'Time (s, Log)']

for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
    ax = axes[i]
    for method in methods:
        config = style_config[method]
        ax.plot(alpha_vals, metric[method], label=method, 
                color=config['color'], marker=config['marker'], 
                linestyle=config['ls'], linewidth=config['lw'], 
                markersize=10 if method == 'FAIR-VUE' else 6, 
                alpha=config['alpha'])
    
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xticks(alpha_vals)
    ax.set_xticklabels(alpha_labels)
    ax.set_xlabel(r"Heterogeneity ($\alpha$)")
    ax.set_ylabel(ylabel)
    
    if 'Time' in title:
        ax.set_yscale('log')
        ax.set_ylim(1, 10000)

# 图例处理
handles, labels = axes[0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
           ncol=7, frameon=True, facecolor='white', edgecolor='none', shadow=True)

plt.savefig("unlearning_robustness_final.pdf", bbox_inches='tight')
plt.show()