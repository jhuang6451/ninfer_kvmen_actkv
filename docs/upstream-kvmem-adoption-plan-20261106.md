# 上游 kvmem（v0.19.1 / master f0a3d51f）可采纳项评估

| 项目 | 内容 |
|---|---|
| 日期 | 2026-11-06 |
| 我方基线 | 本仓库 `a2e97a3`（VERSION 0.17.0），ninfer-all 钉在 `8319e8f`，补丁为单个 `ninfer-kvmem.patch`（1 MB） |
| 上游 | `kvmem/kvmem` master；ninfer-all 钉在 `6cc1f9fa`，补丁为 **40 个线性依赖的小补丁**（`backends/versions.json` 的 `requires` 链） |
| 原则 | 不整体并入上游；只采纳算法与工程上的长处，按我方 ActKV 结构重新实现 |
| 验证状态 | **只读了上游代码与文档，没有编译、没有跑任何模型。** 下面的"收益"都是推断，需要 A/B 验证 |

## 1. 先说结论

1. **补丁无法逐个 cherry-pick。** 上游 ninfer 补丁是一条依赖链（`kvmem → … → content-cache → release-20261104-integration`），
   且基于更新的 ninfer-all。单拿某一个补丁会缺前置。可行路径是"按思想重写 + 手工接线"，
   或者先把 ninfer-all 基线升到 `6cc1f9fa` 再重做我们的补丁（成本最大，建议最后再考虑）。
2. **上游已经从我们这里拿走了不少东西**（`dflash2-gguf-blocks`、`dflash2-nvfp4`、`cuda-graph-probe`、
   `ngram-arbitration`、`dflash-conditioning`，README 也署名 LRFU/GQA max 来自 kvmem-actkv）。这些不用再"采纳"。
3. **上游检索默认方案已离开 "余弦 + LRFU/max"**，改为 *page-softmax + 一跳图传播（alpha=0.2）*，
   LRFU 只在旧的 `meanq-cosine` 下默认启用（`docs/graph-recall.md`）。
   但该文档自己写明未做答案质量基准（"throughput or answer-quality benchmarks" 不在验收范围），
   所以**没有证据表明它在 200K+ 的召回/幻觉上优于我们的 ActKV**。要用我们自己的 33 针测试说话。

## 2. 逐项评估

### P0：成本低、与 ActKV 目标直接相关（建议先做）

| # | 上游做法 | 我方现状 | 建议 |
|---|---|---|---|
| A | **LRFU 负分语义**：命中比例只门控*正*增量，未命中页保留 `min(r,0)`；`H_t = λ·H_{t-1} + min(r,0) + hit·max(r,0)`（`kvmem/include/kvmem/lrfu.hpp` 注释说明：把未命中的负分改成 0 会让它排到"命中但分数略负"的页前面，颠倒相关性） | `accumulate_lrfu`（`kvmem/src/host/kvmem_store.cpp:280`）只累加 `score>0` 且命中（且在本轮选中集内）的块，其余负分/未命中分一律按 0 处理，最后再 clamp 到 0（已核对 `kvmem_store.cpp:326-336`；设计原因见 `docs/actkv-port-implementation-record.md` §3.3）。即负相关证据被整体丢弃 | 改成上游语义，补单测；这是 ~20 行的算法修正。**注意**：我们的 clamp 是为了"冷块能回升"，上游用另一种方式解决，改前要用 33 针对比 |
| B | **LRFU 状态与检查点绑定**：`LrfuHistory` 有 `begin_round / preview / commit`，只在"已接受的 prefill"提交一次，探测与工具续写、推测解码被拒的行都不提交；回退/分叉时按输入边界恢复（`LrfuPoint`、`lrfu_prefix_points`） | 用 block 前沿推断轮次（`round_id==0` 时"上下文变长=新一轮"），ninfer 侧只能这样（见代码注释） | 在编辑、回退、多分支（OpenCode）场景下我们的启发式会重复衰减或漏衰减。借 `lrfu.hpp`（150 行，纯头文件、仅依赖 `snapshot.hpp`）的接口思想重写 |
| C | **有界 Query**：`query_policy.hpp`（70 行）只取最近 `--kvmem-query-max-tokens`（默认 512）个用户 token 做检索，重放另有 2048 上限 | 未核实我们的 Query 取多长。若 200K 长 prefill 的整段输入都参与打分，会稀释信号 | **先读我方 `prepare_window()` 的 Query 范围再决定。** 若确实偏长，这是对"200K+ 幻觉"最便宜的一刀 |

### P1：中等成本，需 A/B 才值得合入

| # | 内容 | 上游规模 | 说明 |
|---|---|---|---|
| D | **page-softmax 评分**：用真实 Q 对 Mean-K 页做缩放点积再 softmax（排除递归层），Q/Mean-K 显式 FP16 边界 | `src/cuda/page_softmax.cu` 256 行 + ninfer `query-memory` 补丁 74 KB + `kvmem-statistics-readback` 30 KB | 思想上等价于 ActKV §7.1（我们 `docs/actkv-compression-in-kvmem.md` 评为"收益高"但未做）。我们已有 `token_sums` op 与 Mean-Q 统计，是一半的基础。以 `--kvmem-query-scorer` 开关实验性接入 |
| E | **一跳图传播**（alpha=0.2，邻接块 + 注意力边，跨消息边系数 0.25，只传正分，负分留在原处） | `graph_recall.cpp` 556 行 + hpp 158 行，依赖 D 的注意力边统计 | 依赖 D；单独做没有意义。负分处理（只传 `max(s,0)`）的思路可直接用于 A |
| F | `--kvmem-score-agg` 默认值 | 上游 meanq-cosine 默认 `max`，page-softmax 默认 `mean` | 我们的启动参数用 `mean`（`ninfer-actkv.container`）。ActKV 论文 Alg.3 用 max，值得做一次 mean/max 对比 |

### P2：工程与稳定性（按需摘取）

小补丁、独立性较好，可按思想手写：

- `output-budget`（4 KB）：自动输出上限受逻辑上下文与 RAM/SSD 可容纳历史约束。
- `token-alignment`（8 KB）：启动时对齐 token 预算再做容量检查，并回报生效值。
- `memory-planning`（27 KB）：启动前预算测算与容量不足时的恢复建议。
- `cpu-vision-props`（7 KB）：CPU 视觉状态查询不抢推理锁（我们用 `--vision-residency cpu`，有实际意义）。
- `monitor-events` / `monitor-image-input`：监控事件、TTFT、完成原因。
- `checkpoint-cleanup`（41 KB）、`input-reuse`（32 KB）：请求后清理与精确输入复用，提高重复请求命中。

### P3：暂缓或不采纳

| 项 | 原因 |
|---|---|
| `checkpoint-policy`(67K) / `chat-cache-owner`(11K) / `cross-owner-fork`(54K) / `shared-cache`(34K) / `content-cache`(81K) | 面向 OpenCode 回退与跨会话分叉，体量大且互相依赖；除非你要用多分支客户端，否则收益不抵成本 |
| `heads-tp`(255K)、`dflash2-layer-pipeline`、`multigpu-resident-vision` | 多 GPU，我们单卡 RTX 5080 |
| `exact-ops`(100K)、`gguf-*mxfp8/nvfp4*`（合计约 400K）、`t5-*` | 预填充性能向，针对 RTX 5060 Ti / GGUF 路径调参；我们 5080 + `.ninfer`，收益未知。若想提速 350 tok/s 的满池 prefill，再单独评估 |
| 原生工具调用约束 | **我方补丁已含**（`tool_grammar.*`、`tool_contract.cpp` 在补丁树中），无需再做 |
| Radiance / PolyStrata / llama.cpp / Windows 打包 / 统一控制台 / Web UI | 与 Linux + NInfer + Podman 路线无关 |

## 3. 建议路线

1. **建立基线**：用当前镜像跑 `niah33_262k.py`、`tests/test_multi_needle.py`，记录 33/33、峰值上下文、prefill 速度
   （2026-10-11 报告：33/33、258,467 token、满池约 350 tok/s）。后面每步都对比它。
2. **读我方 Query 范围**（C 的前置），确认是否值得加上限。
3. **做 A，再做 C（若适用）**：只改 `kvmem/` 核心与 ninfer 的 `serve_options`，每步一个提交、一轮 33 针。
4. **做 B**（LRFU 状态按已接受 prefill 提交）：补针对"编辑/回退/重试"的单测。
5. **实验 D（+E）**：默认关闭，A/B 对比 `legacy`(ActKV) 与 `page-softmax`；不达标就不合入。
6. P2 项按痛点逐个摘取。
7. 最后才考虑是否值得升级 ninfer-all 基线。

## 4. 风险与未知

- 未编译、未运行；上面所有"等价/收益"均来自读文档与代码。
- 上游 LRFU 与我们的 `clamp 到 0` 取舍不同，二者都没有公开的答案质量对照，**必须以 33 针结果为准**。
- 本仓库原有的 `docs/actkv-*` 记录显示：端到端编译/真实模型验证在 2026-10-10 移植时并未完成，
  之后 10-11 的 262K 报告是目前唯一的实测证据。
