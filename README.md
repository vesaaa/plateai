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

## Docker 快速开始

```bash
docker pull ghcr.io/vesaaa/plateai:latest

# 1) 基础自检
docker run --rm ghcr.io/vesaaa/plateai:latest info

# 2) 训练（示例）
# CSV/Excel 约定：
#   第1列 = 车牌真值（如 粤BAE6196）
#   第2列 = 图片URL 或 本地路径
docker run --rm \
  -v $(pwd)/data:/data:ro \
  -v $(pwd)/output:/workspace/output \
  -v $(pwd)/cache:/workspace/cache \
  ghcr.io/vesaaa/plateai:latest \
  train \
    --csv /data/hard.csv \
    --output /workspace/output/plate_rec_color.onnx \
    --epochs 10 \
    --batch-size 32

# 3) 仅导出（不训练）
docker run --rm \
  -v $(pwd)/output:/workspace/output \
  ghcr.io/vesaaa/plateai:latest \
  export \
    --checkpoint /workspace/output/best.pth \
    --output /workspace/output/plate_rec_color.onnx
```

训练出的 `plate_rec_color.onnx` 可直接替换 `platex/models/plate_rec_color.onnx`。

## 输入文件格式（CSV / Excel）

PlateAI 解析规则如下：

- 第 1 列：车牌标签，例如 `粤BAE6196`、`京A12345`。
- 第 2 列：图片 `HTTP(S) URL` 或本地文件路径。
- 第二列若是以 `/` 开头且无协议，会按 `--url-prefix` 进行补全（默认是 platex 示例桶前缀）。
- 若首行不是“车牌样式字符串”，会自动识别为表头并跳过。
- `.csv` 编码按 `utf-8-sig` -> `utf-8` -> `gb18030` -> `gbk` 顺序尝试。
- `.xlsx` 依赖 `openpyxl`（镜像内已包含）。
- 下载图片会缓存到 `/workspace/cache`，建议挂载为持久卷以加速重复训练。

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

## CI/CD 触发说明（重点）

当前仓库的 GitHub Actions 工作流（`.github/workflows/build.yml`）触发条件是：

- `push` 到 `main` 分支：会构建并推送镜像。
- `push` tag（`v*`）：也会构建并推送镜像。
- `workflow_dispatch`：可在 GitHub Actions 页面手动触发。
- `pull_request` 到 `main`：只做构建校验，不推送镜像。

所以现在**不是必须打 tag 才会跑**；你只要提交到 `main`，流水线就会自动触发并更新镜像（含 `latest`）。

## 路线图

- [ ] 小样本场景下引入可选 CCPD 预热数据。
- [ ] 面向高频混淆对的损失加权。
- [ ] 训练结束后内置 held-out CSV 评估报告。
