# 运维脚本（调优 / 基准 / 同步）

在服务器上推荐目录：`/opt/vscc/plateai/tools/`。

| 脚本 | 说明 |
|------|------|
| `sync_tools.sh` | 从 GitHub `main` 拉取本目录全部 `.sh`/`.py`（需 `curl`），无需本地 git 仓库。 |
| `start_autotune.sh` | 启动迭代调优；可选 `autotune.env`（见 `autotune.env.example`）。 |
| `iter_train_platex_loop.sh` | 核心循环：bench → 混合 CSV → Docker 训练 → 验证部署 / 回滚。 |
| `sample_training_pool.py` | 从 10万/20万级 CSV 抽样训练池（建议在 Docker 内执行）。 |
| `restore_optimal_we.sh` | 从 `BEST_EVAL.txt` 或 `OPTIMAL_*` 恢复 WE 权重并重启 platex。 |
| `bench_platex_csv.py` | 固定集评测。 |
| `build_train_mix.py` | 错例 + 正样本池合并。 |

### 首次部署工具

```bash
export PLATEAI_HOME=/opt/vscc/plateai
curl -fsSL -o /tmp/sync_tools.sh \
  https://raw.githubusercontent.com/vesaaa/plateai/main/tools/sync_tools.sh
bash /tmp/sync_tools.sh
```

### 后台启动调优

```bash
cp autotune.env.example autotune.env   # 按需编辑（含 BASELINE_ACC / BASELINE_ONNX / BASELINE_PTH）
export AUTO_BACKGROUND=1
bash /opt/vscc/plateai/tools/start_autotune.sh
```

基线说明：`iter_train_platex_loop.sh` 会在 **`backups/OPTIMAL_*acc_0_926000*`** 存在时，把 **accepted 回滚快照** 初始化为该黄金 WE；bench 只有 **≥ BASELINE_ACC** 才覆盖 `BEST_EVAL`；训练后 verify 需 **同时优于会话最优且优于 BASELINE_ACC** 才保留部署。若整轮结束仍 **低于 BASELINE_ACC**，退出时会 **自动恢复** 基线 ONNX/`best.pth` 并更新 `BEST_EVAL`。

### 大池抽样（宿主机无 plateai 包时）

```bash
docker run --rm --entrypoint python3 \
  -v /opt/vscc/plateai:/ws -w /ws \
  ghcr.io/vesaaa/plateai:v1.0.3-cpu \
  /ws/tools/sample_training_pool.py \
  --input /ws/data/20万原图.csv \
  --output /ws/data/pool_sampled_50k.csv \
  --max-total 50000 --stratify-prefix
```
