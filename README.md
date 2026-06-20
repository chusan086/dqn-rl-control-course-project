# DQN 强化学习控制任务课程项目

本仓库为《人工智能》课程项目，主要复现并比较 Random、DQN 和 Double DQN 在 Gymnasium 离散控制任务上的表现。

## 项目结构

```text
src/                    训练、评估、绘图和工具代码
experiments/logs/       轻量级 CSV/JSON 实验日志
experiments/figures/    由实验日志生成的奖励曲线
paper_assets/figures/   课程论文使用的图
paper_assets/tables/    课程论文使用的汇总表
requirements.txt        Python 依赖列表
REPRODUCIBILITY.md      实验复现说明
```

模型 checkpoint 可以通过训练命令重新生成，因此未提交到仓库。

## 环境配置

实验记录使用 Linux 虚拟机、Python `3.10.12` 和项目本地虚拟环境。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
```

所有命令均从仓库根目录运行。

## 快速检查

```bash
.venv/bin/python src/smoke_cartpole.py --episodes 5 --seed 2026
.venv/bin/python src/train_dqn.py --episodes 20 --algo dqn --log-interval 5 --seed 2026
.venv/bin/python src/train_dqn.py --episodes 20 --algo double_dqn --log-interval 5 --seed 2027
```

## 复现实验结果

CartPole 主实验：

```bash
.venv/bin/python src/run_phase3_cartpole.py --env-id CartPole-v1 --seeds 2026,2027,2028 --episodes 300 --tag phase3 --log-interval 50
.venv/bin/python src/plot_cartpole_results.py --env-id CartPole-v1 --tag phase3
```

LunarLander 短训练扩展实验：

```bash
.venv/bin/python src/run_phase3_cartpole.py --env-id LunarLander-v3 --seeds 2026,2027 --episodes 50 --tag phase4short --log-interval 10
.venv/bin/python src/plot_cartpole_results.py --env-id LunarLander-v3 --tag phase4short
```

更多细节见 `REPRODUCIBILITY.md`。

## 实验结果

CartPole-v1，3 个随机种子，300 episodes：

| 算法 | 平均奖励 | 后 50 回合平均奖励 | 最佳移动平均奖励 |
| --- | ---: | ---: | ---: |
| Random | 22.366 | 22.060 | 29.367 |
| DQN | 151.759 | 97.940 | 410.517 |
| Double DQN | 190.607 | 173.847 | 479.733 |

LunarLander-v3，2 个随机种子，50 episodes：

| 算法 | 平均奖励 | 后 50 回合平均奖励 | 最佳移动平均奖励 |
| --- | ---: | ---: | ---: |
| Random | -168.752 | -168.752 | -111.264 |
| DQN | -21.741 | -21.741 | 32.715 |
| Double DQN | -16.960 | -16.960 | 30.096 |

CartPole 是本文的主要实验依据。LunarLander 作为更复杂任务的短训练扩展，用于讨论算法在更难环境中的表现和局限。
