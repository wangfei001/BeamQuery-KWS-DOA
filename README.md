# BeamQuery — 多麦克风唤醒词检测 + 粗方位（DOA）联合估计

本仓库是 WeKWS 工程，主模型为 **BeamQuery**；相关 recipe 为 `examples/hi_xiaowen/s0/`。

---

## 1. 主模型（BeamQuery）创新点

### 1.1 方位类别结构化绑定的波束 query
每个粗方位类别 `c ∈ {0,1,2}`（`0:[0,60°)`、`1:[60,120°)`、`2:[120,180°)`）拥有一个
专属 query，该 query **在结构上绑定**到本类内由延迟求和（delay-and-sum）波束在类内角度
网格（`beam_step_deg`，最终模型 8°）上算出的波束特征（`query_mode: beam`）。
检测器对**每个候选方向都计算一遍**，而不是先估一个 DOA 再送去检测；方位角直接由"给出
唤醒证据的那个类别索引"读出，不需要单独的角度回归分支。这消除了级联系统的单点失效：
内容无关的 DOA 估计器一旦锁定干扰说话人，其后的单通道检测器就再也看不到唤醒词。

### 1.2 统一的 (Q+1) 分类目标
把类别唤醒判决写成 (Q+1) 类分类：query 索引即方位类别，另加一个"无唤醒"类，其 logit
固定为 0；正样本 target = 唤醒词真实方位类，负样本 target = 无唤醒类。**一个损失同时
完成"检测 + 方位绑定"**，且与推理口径完全一致：

```
det = max_c sigma(S_c) > eta ,      that = argmax_c S_c
```

### 1.3 辅助 presence 头（`lambda_presence`，最终模型 0.1）
每个 query 额外预测"方向 c 上是否存在声源"（presence logit `r_c`），迫使共享表示编码
方向信息、抑制 DOA 表示退化。

### 1.4 分离的辅助唤醒头（detached，`lambda_wake = 0.3`）
低虚警工作点的检测分数由**辅助二分类唤醒头**给出（`use_wake_head: true`），该头读取共享
特征的 **detached 副本**（`wake_detach: true`，stop-gradient），辅助损失不会通过它反向
改写主分支特征。

### 1.5 训练—推理一致的严格流式
前端帧局部化（`causal: true`），归一化用 running 统计量（`norm_type: running`），训练阶段
直接使用流式解码器所用的 running class-max 统计量（`pool_type: run_max_logit`），检测与
定位走同一条推理路径。

### 1.6 输入特征
类内波束响应的**幅度**与相干比，加上麦克风对 **TDOA->sin/cos** 特征
（`pair_feat_mode: tdoa`、`use_pair_ipd: true`），避免 8 cm 孔径下高频相位卷绕。

---

## 2. 主模型代码与运行

### 2.1 目录

| 路径 | 内容 |
|---|---|
| `examples/hi_xiaowen/s0/paper_experiments/main_model/` | **主模型代码**：`model.py`、`batched_forward.py`、`streaming.py`、`train.sh`、`eval.py`、`run_streaming.py`、`config.yaml`、`configs/stage{1,2}.yaml` |
| `examples/hi_xiaowen/s0/conf/train_dynamic_query.yaml` | 主模型 + 阵列 + RIR 仿真配置 |
| `examples/hi_xiaowen/s0/run.sh` | 数据准备 / 多通道仿真 / 主模型训练与评测入口 |
| `examples/hi_xiaowen/s0/tools/` | `gen_multi_channel_wav.py`、`append_doa_to_list.py`、`check_room_rir.py` 等 |
| `examples/hi_xiaowen/s0/local/` | Mobvoi Hotwords 语料下载与清单生成 |
| `examples/hi_xiaowen/s0/wekws/` | 主模型运行时依赖的框架副本（`model/spatial_dynamic_query.py`、`utils/room_rir.py`、`dataset/spatial_kws_dataset.py` 等） |
| `wekws/`、`tools/` | WeKWS 框架与通用工具 |

### 2.2 环境

```bash
cd examples/hi_xiaowen/s0
. ./path.sh
pip install -r ../../../requirements.txt
```

### 2.3 数据生成

```bash
download_dir=data/download bash run.sh -1 -1     # 下载并解压 Mobvoi Hotwords（约 1.5 GB）
bash run.sh 0 1                                  # wav.scp / text / wav.dur / data.list + CMVN
bash run.sh 5 5                                  # 多通道 + 房间 RIR 仿真
bash run.sh 5 5 --rir_mode direct                # 无混响直达声版本（消融）
python tools/check_room_rir.py --config conf/train_dynamic_query.yaml   # 校验 RIR 与方位约定
```

> `run.sh` 的 **stage 号必须写在最前**，选项在后：
> `bash run.sh 5 5 --rir_mode direct` OK ／ `bash run.sh --rir_mode direct 5 5` 错

房间/阵列/RIR 参数见 `conf/train_dynamic_query.yaml` 的 `data_gen.room_rir` 段（房间
1.65x4.02x1.32 m、双麦线阵、`t60_range`、`source_distance_range`、`nsample`、`order`，
后端 `rir_generator` / `pyroomacoustics` / `roomsimove`）。

### 2.4 训练

```bash
bash run.sh 6 6                                  # recipe 入口 -> exp/spatial_model/best_cv.pt
bash paper_experiments/main_model/train.sh       # 主模型两阶段训练（推荐）
```

两阶段流程：`configs/stage1.yaml`（`mean_max` 池化）-> 以 `stage1/best_cv.pt` 初始化
`configs/stage2.yaml`（`run_max_logit`）-> 复制为 `experiment/best_cv.pt`。

### 2.5 评测

```bash
bash run.sh 7 7                                  # 用 best_cv.pt 重评

python paper_experiments/main_model/eval.py      # 全测试集 -> test_metrics.json
python paper_experiments/main_model/run_streaming.py   # 严格流式（逐帧解码、RTF、时延）
```

`eval.py` 输出 AUC / FRR@1/h / 全正样本 DOA 准确率 / MAE / Joint；
`run_streaming.py` 输出 `streaming_results.json`。
