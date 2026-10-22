# PlateAI

`PlateAI` 是一个面向 `platex` 的小型增量训练工具：输入你整理好的失败样本（CSV/Excel），在 `we0091234` 的 `plate_rec_color` 预训练权重上做微调，导出新的 `plate_rec_color.onnx`，再替换到 `platex` 中循环迭代。

建议的闭环流程：

```text
platex 在线识别 -> 收集识别失败样本 -> 人工修正真值标签
      ↓
将样本喂给 plateai 训练 -> 导出新的 ONNX
      ↓
替换 platex/models/plate_rec_color.onnx -> 重启服务 -> 复测
```

## 官方 Release 权重（WE，与 platex Docker 默认一致）

固定版本（推荐用于可复现部署与训练接力）：

- 部署用 ONNX：`https://github.com/vesaaa/plateai/releases/download/v2026.5/plate_rec_color.onnx`
- 训练接力用 checkpoint：`https://github.com/vesaaa/plateai/releases/download/v2026.5/best.pth`

跟随 GitHub **Latest** 标签（文件名不变时会自动指向当前最新 release；缓存层可能更少失效）：

- `https://github.com/vesaaa/plateai/releases/latest/download/plate_rec_color.onnx`
- `https://github.com/vesaaa/plateai/releases/latest/download/best.pth`

下载 `best.pth` 后，可通过 `--pretrained /path/to/best.pth`（或 `PLATEAI_PRETRAINED`）在上一轮权重上继续微调。

## Docker 快速开始

```bash
docker pull ghcr.io/vesaaa/plateai:cpu

# 1) 基础自检
docker run --rm ghcr.io/vesaaa/plateai:cpu info

# 2) 训练（示例）
# CSV/Excel 约定：
#   第1列 = 车牌真值（如 粤BAE6196）
#   第2列 = 图片URL 或 本地路径
docker run --rm \
  -v $(pwd)/data:/data:ro \
  -v $(pwd)/output:/workspace/output \
  -v $(pwd)/cache:/workspace/cache \
  ghcr.io/vesaaa/plateai:cpu \
  train \
    --csv /data/hard.csv \
    --output /workspace/output/plate_rec_color.onnx \
    --epochs 10 \
    --batch-size 32

# 3) 仅导出（不训练）
docker run --rm \
  -v $(pwd)/output:/workspace/output \
  ghcr.io/vesaaa/plateai:cpu \
  export \
    --checkpoint /workspace/output/best.pth \
    --output /workspace/output/plate_rec_color.onnx
```

训练出的 `plate_rec_color.onnx` 可直接替换 `platex/models/plate_rec_color.onnx`。

镜像标签约定（按发布版本自动生成）：
- CPU：`ghcr.io/vesaaa/plateai:vX.Y.Z-cpu` 与 `ghcr.io/vesaaa/plateai:cpu`
- CUDA11：`ghcr.io/vesaaa/plateai:vX.Y.Z-cuda11` 与 `ghcr.io/vesaaa/plateai:cuda11`
- CUDA12：`ghcr.io/vesaaa/plateai:vX.Y.Z-cuda12` 与 `ghcr.io/vesaaa/plateai:cuda12`

GPU 训练提示：
- 使用 `cuda11/cuda12` 镜像时，请为容器开启 GPU（例如 `docker run --gpus all ...`）。
- 训练设备仍建议显式传 `--device cuda`（或 `PLATEAI_DEVICE=cuda`）。

## 输入文件格式（CSV / Excel）

PlateAI 解析规则如下：

- 第 1 列：车牌标签，例如 `粤BAE6196`、`京A12345`。
- 第 2 列：图片 `HTTP(S) URL` 或本地文件路径。
- 第二列若是以 `/` 开头且无协议，会按 `--url-prefix` 进行补全（默认是 platex 示例桶前缀）。
- 若首行不是“车牌样式字符串”，会自动识别为表头并跳过。
- `.csv` 编码按 `utf-8-sig` -> `utf-8` -> `gb18030` -> `gbk` 顺序尝试。
- `.xlsx` 依赖 `openpyxl`（镜像内已包含）。
- 下载图片会缓存到 `/workspace/cache`，建议挂载为持久卷以加速重复训练。

`--url-prefix` 使用说明（非常重要）：

- 当 CSV 第 2 列是完整 URL（`http://` 或 `https://`）时：**不需要** `--url-prefix`。
- 当 CSV 第 2 列是以 `/` 开头的相对路径时：**必须**传 `--url-prefix`，用于拼接成完整下载地址。
- 示例：第二列是 `/SNTDA-500-LS19030650/.../plate.bmp`，则可传：
  `--url-prefix https://huizhoupark.obs.cn-south-1.myhuaweicloud.com`
- 若频繁出现 `403`：
  - 先确认 `--url-prefix` 是否正确；
  - 再确认对象存储是否有防盗链/鉴权限制（部分路径可能需要签名 URL）；
  - `403` 样本会被跳过，不会阻塞整轮训练，但会减少有效样本数。

最小示例：

```csv
plate,image
粤BAE6196,https://example.com/imgs/0001.jpg
粤LDD7691,/SNTDA-500-LS19030650/.../plate.bmp
```

可参考：`examples/sample.csv`。

## 注意事项（强烈建议先看）

- 不要从零训练：默认从 `weights/plate_rec_color.pth` 起步，避免小样本过拟合导致泛化崩掉。
- 优先保证标签质量：错标样本会直接把模型带偏，宁可少也要准。
- 每轮小步快跑：建议先 5~10 epoch 做快速验证，再决定是否加大训练量。
- 保留 `cache` 挂载：重复下载图片会拖慢训练且增加外部依赖不稳定性。
- 训练后必须回测：以你固定的 1000 张（或同分布）数据集做对比，观察准确率和耗时。
- 上线前保留回滚：替换 ONNX 时保留旧文件，确保可以秒级回退。

## 新手必看：一批错误图可以训练几次？

可以训练很多次，不是只能训练 1 次。

- 一次长训：同一批数据把 `--epochs` 调大，一次跑完。
- 多轮接力（推荐）：每轮训练后把 `best.pth` 保存下来，下一轮用 `--pretrained` 接着训。

关键点：

- `plate_rec_color.onnx`：部署文件（给 platex 用）。
- `best.pth`：训练接力文件（给下一轮训练继续用）。
- 默认镜像里的 `/workspace/weights/plate_rec_color.pth` 不会自动被你训练覆盖。
- 每次训练会在 `--checkpoint-dir` 目录产出新的 `best.pth`（默认路径：`/workspace/checkpoints/best.pth`）。
- 如果不把 `--checkpoint-dir` 挂载到宿主机持久目录，容器退出后该 `best.pth` 会丢失，下一轮就无法直接接力。

建议每轮都保存：

- `output/plate_rec_color.onnx`
- `checkpoints/best.pth`（建议改名备份，如 `best_20260429_round1.pth`）

Docker 示例（推荐加上 checkpoints 挂载）：

```bash
docker run --rm \
  -v $(pwd)/data:/data:ro \
  -v $(pwd)/output:/workspace/output \
  -v $(pwd)/cache:/workspace/cache \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  ghcr.io/vesaaa/plateai:latest \
  train \
    --csv /data/hard.csv \
    --checkpoint-dir /workspace/checkpoints \
    --output /workspace/output/plate_rec_color.onnx
```

下一轮接力训练示例：

```bash
plateai train \
  --csv /data/hard_next.csv \
  --pretrained /workspace/checkpoints/best_20260429_round1.pth \
  --checkpoint-dir /workspace/checkpoints \
  --output /workspace/output/plate_rec_color.onnx
```

## 常用参数说明（小白版）

- `--pretrained`：从哪个 `.pth` 开始训练。  
  不传则用镜像内置底座；想“接着上次继续练”就传上轮 `best.pth`。
- `--epochs`：训练轮数。  
  轮数越大训练越久，不代表一定更好，建议每轮训完就回测。
- `--lr`：学习率。  
  数值越大改动越猛，越小越稳。接力训练时建议逐轮减小。
- `--batch-size`：每批样本数。  
  越大越快，但吃内存更多；内存不够就调小。
- `--hard-case-repeat`：困难样本重复倍数。  
  小样本时建议保留默认 `4`，让模型更关注错误样本。
- `--val-ratio`：验证集比例。  
  样本太少时可适当调大（如 `0.15~0.2`）让验证更稳。

参数优先级规则：

- 命令行参数（`--xxx`）优先级最高。
- 未传命令行参数时，读取对应环境变量（ENV）。
- ENV 也没有时，使用程序默认值。

常用 ENV（与命令行一一对应）：

- `PLATEAI_DEVICE` -> `--device`（`cpu` / `cuda`）
- `PLATEAI_EPOCHS` -> `--epochs`
- `PLATEAI_BATCH_SIZE` -> `--batch-size`
- `PLATEAI_LR` -> `--lr`
- `PLATEAI_WORKERS` -> `--workers`
- `PLATEAI_VAL_RATIO` -> `--val-ratio`
- `PLATEAI_HARD_CASE_REPEAT` -> `--hard-case-repeat`
- `PLATEAI_PRETRAINED` -> `--pretrained`
- `PLATEAI_CHECKPOINT_DIR` -> `--checkpoint-dir`
- `PLATEAI_CACHE_DIR` -> `--cache-dir`
- `PLATEAI_SEED` -> `--seed`
- `PLATEAI_MAX_ROWS` -> `--max-rows`（`0` 表示不限）

示例（手动传参 + ENV 混用）：

```bash
docker run --rm \
  -e PLATEAI_DEVICE=cuda \
  -e PLATEAI_BATCH_SIZE=16 \
  -e PLATEAI_WORKERS=1 \
  -v $(pwd)/data:/data:ro \
  -v $(pwd)/output:/workspace/output \
  -v $(pwd)/cache:/workspace/cache \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  ghcr.io/vesaaa/plateai:latest \
  train \
    --csv /data/hard.csv \
    --epochs 6 \
    --lr 3e-4 \
    --output /workspace/output/plate_rec_color.onnx
```

## 推荐训练节奏（以小批 hard case 为例）

以“约 200~500 条失败样本”为例，可以这样跑：

1) 第一轮（快速起步）

```bash
plateai train \
  --csv /data/hard_round1.csv \
  --output /workspace/output/plate_rec_color.onnx \
  --epochs 10 \
  --lr 5e-4
```

2) 第二轮（接着训，降学习率）

```bash
plateai train \
  --csv /data/hard_round2.csv \
  --pretrained /workspace/checkpoints/best_round1.pth \
  --output /workspace/output/plate_rec_color.onnx \
  --epochs 5 \
  --lr 2e-4
```

3) 第三轮（小步微调）

```bash
plateai train \
  --csv /data/hard_round3.csv \
  --pretrained /workspace/checkpoints/best_round2.pth \
  --output /workspace/output/plate_rec_color.onnx \
  --epochs 3 \
  --lr 1e-4
```

如果连续两轮在固定基准集（如 1000/2000 张）都没有提升，就建议停止当前参数组合，换新样本再训，避免过拟合。

## 本地开发（不走 Docker）

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[torch]"

plateai info
plateai train --csv examples/sample.csv --output output/plate_rec_color.onnx --epochs 2
```

默认预训练权重：`weights/plate_rec_color.pth`。如需指定其它初始权重，可传 `--pretrained`。

## 8C16G CPU 机器的经验配置

每次训练都基于预训练权重微调（不是从零开始）。在 8C16G、CPU-only 环境下，可参考：

- 样本约 100：10 epoch，通常几分钟。
- 样本约 1,000：10 epoch，通常 30~60 分钟。
- 样本约 10,000：10 epoch，通常数小时。

`--hard-case-repeat` 会对疑难样本重复采样，小数据集时可提升训练信号强度。

## 在 platex 中验证新模型

```bash
cp output/plate_rec_color.onnx /path/to/platex/models/plate_rec_color.onnx
docker restart platex
```

然后执行你的基准测试。若加载成功，`platex` 日志中通常会看到：

- `Dual model session pool initialized`（对应 `plate_rec_color.onnx`）
- `WE recognizer loaded`

## 全量训练新识别模型（PlateCRNN，实验中）

`plate_rec_color` 只有 21 个 CTC 时间步，8 位新能源车牌漏位、连续重复字符（`DD`/`44`）被合并都是结构性问题，靠错例微调无法消除。`plateai/crnn/` 提供从头训练的新识别模型：

- 输入 `32×160` BGR，`(x/255-0.5)/0.5`；CNN + BiLSTM，输出 `40` 个 CTC 时间步，字表 = `plate_rec_color` 字表 + `应急`（81 类，0 为 blank）。
- 0.73M 参数，ONNX 约 2.9MB，单线程 CPU 推理约 2.5ms。

流程（以 20 万人工标注小图为例）：

```bash
# 1) 把图片镜像到本地（只下载缺失的）
python3 tools/fetch_images.py data/plate_20w.csv data/img 128

# 2) 预处理 + 按车牌号切分（同一车牌不会同时出现在训练和测试集）
python3 -m plateai.crnn.prep --csv data/plate_20w.csv --img-dir data/img --out data/arr32x160

# 3) 训练（有 GPU 会自动用 cuda）
python3 -m plateai.crnn.train --data data/arr32x160 --out runs/crnn_s --epochs 8

# 4) 导出 ONNX 并评测；可选：同一测试集上 platex 的跑分做对比
python3 tools/bench_platex_local.py data/arr32x160/test_split.csv data/img full data/test_full.jsonl
python3 -m plateai.crnn.evaluate --run runs/crnn_s --data data/arr32x160 --platex-bench data/test_full.jsonl
```

`test_split.csv` 可由 `paste -d, test_labels.txt test_paths.txt` 生成。错例分析：`tools/analyze_bench_jsonl.py`（按错误类型归类）、`tools/error_sheet.py`（错例拼图，人工核对标注）。

基线（2026-10-22，20 万集按车牌号留出 20,082 张测试，12 核 CPU 训练 8 轮约 2.5 小时）：

| | platex v1.0.11（full） | PlateCRNN |
|---|---|---|
| 整体 | 86.5% | **93.5%** |
| 7 位 / 8 位新能源 | 88.0% / 84.2% | 92.5% / 95.1% |
| 粤 / 非粤 | 87.9% / 61.1% | 96.3% / 43.9% |
| 置信度 ≥ 0.7 时 | — | 覆盖 92.5%，准确率 98.1% |

### 标注清洗（`plateai.crnn.score` + `plateai.crnn.clean`）

```bash
python3 -m plateai.crnn.score --run runs/crnn_s --data data/arr32x160 --split train --out data/train_scored.jsonl
python3 -m plateai.crnn.clean --data data/arr32x160 --scored data/train_scored.jsonl \
    --trusted data/plate_mini_10w.csv --out data/arr32x160_clean
```

规则：`--trusted`（人工清洗过的 CSV）中的行全部保留；被人工清洗覆盖 ≥80% 的日期里、未进入 trusted 的行视为人工剔除；其余行在模型结果与标注编辑距离 ≥2 时剔除（干净数据上模型自身错误 88% 只差 1 个字，错标通常差多个字）。

2026-10-22 实验（20 万集，用户清洗的 10w 作为 trusted；人工抽查确认被剔除行多为错标）：训练集 179,805 → 172,557。在 9,991 张干净留出测试图（留出集 ∩ 10w）上：

| | 原模型 | 清洗后重训 |
|---|---|---|
| CRNN 单独 | 96.79% | 96.94% |
| platex auto（阈值 0.7） | 98.80% | 98.69% |
| 最优阈值融合（离线模拟） | 98.25% @0.7 | 98.25% @0.6 |

结论：3–7% 的错标对训练影响很小，清洗后重训端到端无提升，未替换已发布权重。干净测试集上剩余约 1.2% 错误主要是外省 / `警` `学` 等稀有字符样本太少，以及模糊、截断的图片；下一步应补充稀有省份数据（或合成数据）并在 GPU 上训更大的模型。

**评测提醒**：20 万集本身约有 7% 错标，评测请用干净的留出集（上表口径），否则会低估 2–3 个点（同一版本在含错标的完整测试集上为 95.6%，在干净集上为 98.8%）。

已知短板：非粤省份样本太少（训练集中粤牌占 94%），剩余错误约一半是省份字；剩余错误中约 15–20% 为标注错误。权重发布在 release `crnn-2026.10-poc`（`plate_rec_crnn.onnx` / `best.pt`），该 tag 不以 `v` 开头，不会触发镜像构建。

## CI/CD 触发说明（重点）

当前仓库的 GitHub Actions 工作流（`.github/workflows/build.yml`）触发条件是：

- 仅 `push` tag（`v*`）触发构建与推送。
- 每次发布会同时推送 `cpu/cuda11/cuda12` 三套镜像标签（含版本标签与通道标签）。

## 路线图

- [ ] 小样本场景下引入可选 CCPD 预热数据。
- [ ] 面向高频混淆对的损失加权。
- [ ] 训练结束后内置 held-out CSV 评估报告。
