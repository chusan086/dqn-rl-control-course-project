# 论文资产

本目录包含课程论文中使用的图和表。

## 图

- `figures/cartpole_phase3_moving_average.png`：CartPole 主实验奖励曲线，按随机种子聚合并显示波动范围。
- `figures/cartpole_phase3_per_seed.png`：CartPole 各随机种子的学习曲线。
- `figures/lunarlander-v3_phase4short_moving_average.png`：LunarLander 短训练扩展实验奖励曲线。
- `figures/lunarlander-v3_phase4short_per_seed.png`：LunarLander 各随机种子的学习曲线。

## 表

- `tables/cartpole_phase3_summary.csv`：CartPole 主实验汇总结果。
- `tables/cartpole_phase3_summary.md`：便于论文写作使用的 Markdown 版本。
- `tables/cartpole_phase3_per_run.csv`：CartPole 各随机种子的单次运行结果。
- `tables/lunarlander-v3_phase4short_summary.csv`：LunarLander 扩展实验汇总结果。
- `tables/lunarlander-v3_phase4short_summary.md`：便于论文写作使用的 Markdown 版本。
- `tables/lunarlander-v3_phase4short_per_run.csv`：LunarLander 各随机种子的单次运行结果。

## 论文定位

CartPole-v1 是主要可复现实验。LunarLander-v3 是更复杂任务上的短训练扩展，由于每个随机种子只训练 50 episodes，因此更适合用于扩展分析和局限讨论。
