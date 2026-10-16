# wekws_code

BeamQuery：多麦克风唤醒词检测 + 粗方位（DOA）联合估计。


---

## 1. 主模型（BeamQuery）创新点

### 1.1 方位类别结构化绑定的波束 query
每个粗方位类别 `c ∈ {0,1,2}`（`0:[0,60°)`、`1:[60,120°)`、`2:[120,180°)`）拥有一个
专属 query，该 query **在结构上绑定**到本类内由延迟求和（delay-and-sum）波束在类内角度
网格上计算出的波束特征（`query_mode: beam`、`beam_step_deg`）。
检测器对**每个候选方向都计算一遍**，而不是先估一个 DOA 再送去检测；方位角直接由"给出
唤醒证据的那个类别索引"读出，不需要单独的角度回归分支。这消除了级联系统的单点失效：
内容无关的 DOA 估计器一旦锁定干扰说话人，其后的单通道检测器就再也看不到唤醒词。

### 1.2 统一的 (Q+1) 分类目标
把类别唤醒判决写成 (Q+1) 类分类：query 索引即方位类别，另加一个"无唤醒"类，其 logit
固定为 0。正样本 target = 唤醒词真实方位类，负样本 target = 无唤醒类。**一个损失同时
完成"检测 + 方位绑定"**，且与评测口径完全一致：


---

## 2. 代码结构、数据生成与实验流程

### 2.1 关键目录

| 路径 | 内容 |
|---|---|
| `wekws/` | WeKWS 框架核心（`model/`、`bin/train.py`、`bin/score.py` 等） |
| `examples/hi_xiaowen_duolu/s0_deepseek_3/` | **本文项目主目录** |
| `examples/hi_xiaowen_duolu/s0_deepseek_3/run.sh` | 数据准备 / 仿真 / 训练 / 评测统一入口 |
| `.../conf/train_dynamic_query.yaml` | BeamQuery 模型 + 阵列 + RIR 仿真配置（单一真值来源） |
| `.../tools/gen_multi_channel_wav.py` | 多通道（房间 RIR）数据生成脚本 |
| `.../local/` | Mobvoi Hotwords 语料下载与清单生成 |
| `.../paper_experiments/` | **论文表格级代码**，每个表一个目录 |
| `.../paper_experiments/main_model/` | 主模型代码：训练 / 评测 / 流式入口 |
| `.../papaer/main_cssp.tex` | 论文源文件（单文件，可直接编译） |

### 2.2 环境

```bash
cd wekws_code/examples/hi_xiaowen_duolu/s0_deepseek_3
. ./path.sh                      # 设置 PYTHONPATH
pip install -r ../../../requirements.txt
```

### 2.3 数据生成

**(a) 下载并准备原始语料（Mobvoi Hotwords，约 1.5 GB）**

```bash
# stage -1：下载并解压到 data/download/
download_dir=data/download bash run.sh -1 -1

# stage 0-1：生成 wav.scp / text / wav.dur / data.list，并计算 CMVN
bash run.sh 0 1
```

**(b) 多通道 + 房间 RIR 仿真（stage 5）**

```bash
# 生成 data/spatial/{train,cv,test}_wav 及 train/cv/test_varcol.tmp.list、*_spatial.list
bash run.sh 5 5

# 仅生成"自由场直达声"数据（无混响，供消融复现）
bash run.sh 5 5 --rir_mode direct

# 校验 RIR 方位约定与物理量（T60 / DRR / 通道间相干 / GCC-PHAT 误差）
python tools/check_room_rir.py --config conf/train_dynamic_query.yaml
```

房间/阵列/RIR 参数全部在 `conf/train_dynamic_query.yaml` 的 `data_gen.room_rir` 段：
房间 `1.65 × 4.02 × 1.32 m`、2 麦线阵、间距 `mic_spacing`（默认 3.3 cm）、
`t60_range`、`source_distance_range`、`nsample`、`order`、RIR 后端
（`rir_generator` / `pyroomacoustics` / `roomsimove`）等。

**(c) 打包为训练用数组（本包未含，需从服务器拷贝或在本地重跑）**

| 目录 | 内容 |
|---|---|
| `data_packed_v5/` | 打包后的 test / eval 多通道数组（`.npy` + meta） |
| `data_packed_v5_largeval/` | 更大验证集版本，主模型训练/评测使用 |
| `data_v5/test_wav/` | 测试集原始波形（最终评测器使用） |
| `data_val10k/` | 1 万条验证集（`eval_wav/` + `*_spatial.list`） |
| `data_external/speech_commands/` | 外部语料（Speech Commands `marvin`，表 6–8 右半部分） |

### 2.4 训练

```bash
# 方式一：recipe 默认单阶段训练（→ exp/spatial_model/best_cv.pt，每轮自动 cv 评测）
bash run.sh 6 6

# 方式二：论文最终配置的两阶段训练（Δ=8°，λ_wake=0.3，3 s 定长协议）
bash paper_experiments/main_model/train.sh            # → paper_experiments/main_model/experiment/
```



各表对应的代码位置：

| 论文表 | 目录 |
|---|---|
| 表 1 语料 | `paper_experiments/table1_corpus/` |
| 表 2 对比系统 | `paper_experiments/table2_systems/` |
| 表 3 波束网格 | `paper_experiments/table3_angle/` |
| 表 4 λ 权重 | `paper_experiments/table4_lambda/` |
| 表 5 消融 | `paper_experiments/table5_ablation/` |
| 表 6 检测 | `paper_experiments/table6_detection/` |
| 表 7 全正样本定位 | `paper_experiments/table7_localisation/` |
| 表 8 检出后定位诊断 | `paper_experiments/table8_localisation_detected/` |
| 表 9 阈值扫描 | `paper_experiments/table9_threshold/` |
| 表 10 流式 | `paper_experiments/table10_streaming/` |
| 外部语料验证 | `paper_experiments/external_validation/` |
| 表格生成 / 校验 | `paper_experiments/table_builder/` |


```
