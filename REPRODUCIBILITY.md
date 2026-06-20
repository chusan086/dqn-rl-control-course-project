# 复现说明

本项目在 Linux 虚拟机和独立 Python 虚拟环境中开发、运行。以下命令默认从仓库根目录执行：

```bash
cd dqn-rl-control-course-project
```

## 环境配置

创建项目本地虚拟环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
```

实验记录环境为 Python `3.10.12`，主要依赖包括 PyTorch CPU、Gymnasium `1.3.0`、NumPy、Pandas、Matplotlib、Seaborn，以及 LunarLander 所需的 Box2D。

## 快速检查

```bash
.venv/bin/python src/smoke_cartpole.py --episodes 5 --seed 2026
.venv/bin/python src/train_dqn.py --episodes 20 --algo dqn --log-interval 5 --seed 2026
.venv/bin/python src/train_dqn.py --episodes 20 --algo double_dqn --log-interval 5 --seed 2027
```

## CartPole 主实验

```bash
.venv/bin/python src/run_phase3_cartpole.py --env-id CartPole-v1 --seeds 2026,2027,2028 --episodes 300 --tag phase3 --log-interval 50
.venv/bin/python src/plot_cartpole_results.py --env-id CartPole-v1 --tag phase3
```

预期输出：

- `experiments/logs/CartPole-v1_phase3_*`
- `experiments/figures/cartpole_phase3_moving_average.png`
- `experiments/figures/cartpole_phase3_per_seed.png`
- `paper_assets/tables/cartpole_phase3_summary.csv`

## LunarLander 扩展实验

```bash
.venv/bin/python src/run_phase3_cartpole.py --env-id LunarLander-v3 --seeds 2026,2027 --episodes 50 --tag phase4short --log-interval 10
.venv/bin/python src/plot_cartpole_results.py --env-id LunarLander-v3 --tag phase4short
```

预期输出：

- `experiments/logs/LunarLander-v3_phase4short_*`
- `experiments/figures/lunarlander-v3_phase4short_moving_average.png`
- `experiments/figures/lunarlander-v3_phase4short_per_seed.png`
- `paper_assets/tables/lunarlander-v3_phase4short_summary.csv`

## 文件保留策略

CSV/JSON 日志、图和论文表格作为轻量级实验证据提交到仓库。模型 checkpoint 会写入 `experiments/checkpoints/`，但未提交，因为它们体积可能较大，并且可以通过训练命令重新生成。
