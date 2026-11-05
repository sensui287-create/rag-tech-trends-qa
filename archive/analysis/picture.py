"""Historical fixed-score radar plot; values are illustrative, not recomputed.

Run explicitly with python -m archive.analysis.picture after installing
requirements-archive.txt. Importing this module does not load a GUI or plot.
"""


def main():
    import matplotlib.pyplot as plt
    import numpy as np

    categories = ['Comprehensiveness', 'Logicality', 'Empowerment', 'Accuracy', 'Overall Winner']
    N = len(categories)

    val_cot = [66, 61, 60, 65, 65]
    val_naive = [51, 53, 56, 59, 56]
    val_tie = [50, 50, 50, 50, 50]

    val_cot += val_cot[:1]
    val_naive += val_naive[:1]
    val_tie += val_tie[:1]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    plt.rcParams['font.family'] = 'serif'

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.set_theta_offset(np.pi / 2) # 将起始位置设在正上方 (12点钟方向)
    ax.set_theta_direction(-1)     # 设置顺时针方向绘制标签 (使得 Logicality 在右侧)

    ax.plot(angles, val_tie, color='#666666', linewidth=2.5, linestyle=':', label='Tie (50% Win Rate)')

    ax.plot(angles, val_cot, color='#1f77b4', linewidth=2.5, linestyle='-', marker='o', markersize=8, label='Agentic RAG (vs CoT)')
    ax.fill(angles, val_cot, color='#1f77b4', alpha=0.15) # 填充内部，设置透明度

    ax.plot(angles, val_naive, color='#d62728', linewidth=2.5, linestyle='-', marker='s', markersize=8, label='Agentic RAG (vs NaiveRAG)')
    ax.fill(angles, val_naive, color='#d62728', alpha=0.15) # 填充内部，设置透明度

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', pad=18) # 增加标签与圆圈的距离，防止重叠

    ax.set_ylim(0, 100)
    y_ticks = [20, 40, 50, 60, 80, 100]
    y_labels = ['20', '40', '50', '60', '80', '100%']

    ax.set_rgrids(y_ticks, labels=y_labels, angle=45, color='grey', size=11)

    ax.grid(color='lightgrey', linestyle='--', linewidth=0.8)

    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(1.2)

    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 0.95), frameon=False, fontsize=13)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
