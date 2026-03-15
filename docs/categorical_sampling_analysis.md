# Categorical 采样与训练前期局部最优分析

## 1. 当前实现

- **采样方式**：`dist = Categorical(logits=logits)`，`action = dist.sample()`，即对 **未做温度缩放** 的 logits 做 softmax 后按概率采样。
- **探索激励**：仅依赖 `ENTROPY_COEF=0.01` 的熵正则。

## 2. 为何容易在前期陷入局部最优

### 2.1 无温度 → 分布很快变尖

- 对 logits 直接 softmax 时，若有效 logits 差异不大（例如 0.15 vs -0.05），概率已会明显偏向某一动作。
- 终端里可见：有效 logits 多在 ±0.1～0.2，某几个 token 略高就会在 softmax 下占主导；mask 用 -1e9，合法动作之间相对差异被放大。
- 结果：**前期** 一旦某些动作因初始化或偶然高 reward 获得略高 logits，分布会很快变尖，探索迅速下降。

### 2.2 正反馈与 “赢家通吃”

- REINFORCE 的梯度是 `log π(a|s) * A`：高 reward 轨迹上的动作会被加强。
- 若早期少数公式（或 token 组合）拿到较高 reward，这些动作的 logits 被推高 → 下次采样更常选到它们 → 更多高 reward 样本 → 梯度继续加强这些动作。
- 没有 PPO 的 clip 或 KL 约束，**策略可以一步内大幅向当前高 reward 区域集中**，容易形成“赢家通吃”和局部最优。

### 2.3 熵正则相对不足

- `ENTROPY_COEF=0.01` 与 policy loss、value loss 量级相比往往偏小；前期 policy gradient 方差大，熵项难以在早期有效拉高探索。
- 若合法动作数较多（vocab 52，每步合法子集仍不小），0.01 的熵系数对“保持较平坦分布”的力度有限，难以单独阻止分布过早变尖。

### 2.4 与终端现象的对应

- 前期很快出现 “New King” 且 BestScore 从 -30 跳到 0.97，说明少数轨迹被强烈强化。
- `legal cnt/bs` 约 98%：多数为合法公式，问题主要在“选哪些合法动作”，而不是合法/非法比例。
- 打印的 logits 中，每步只有少数位置非 -1e9，且在这些位置里常有一个明显较大的 logit（如 0.15～0.2），对应 **Categorical 下该 token 概率远高于其它**，与“前期就偏向少数动作”一致。

## 3. 结论

- **是的**：在当前设置下（无温度、REINFORCE、仅 0.01 熵正则），**Categorical 采样容易在训练前期就让策略分布变尖，从而较早陷入局部最优**。
- 机制主要是：无温度 + 无约束的策略更新 + 相对弱的熵项 → 早期高 reward 动作被快速放大 → 探索不足。

## 4. 缓解思路（建议）

1. **温度调度（推荐）**  
   - 采样时使用 `logits / temperature`，`temperature > 1` 时更平坦。  
   - 例如：前期 `temperature=1.5～2.0`，随 step 线性或指数衰减到 1.0，既保证前期探索，又不过度影响后期收敛。

2. **提高前期熵系数或熵目标**  
   - 如 `ENTROPY_COEF` 从 0.05～0.1 起步，随 step 衰减到 0.01；或设最小熵目标（target entropy）用 PID 控制。

3. **对合法动作做 ε-greedy**  
   - 以小概率 ε（如 0.05～0.1）在合法动作中均匀随机选，其余用 Categorical 采样，直接增加探索。

4. **PPO 或 KL 约束**  
   - 引入 clip 或 KL(π_old, π_new) 上限，限制单步更新幅度，减缓分布骤然变尖。

5. **Curriculum / 先验**  
   - 前期对部分步或部分位置使用更均匀的先验（如合法动作上的均匀分布），再逐渐过渡到纯策略采样。

若只做最小改动，**在 engine 里为 Categorical 采样加上可调的温度（并做简单前期高、后期低的调度）** 即可明显缓解“前期就陷入局部最优”的现象。

---

## 5. 已实现的缓解（本仓库）

- **`ModelConfig.SAMPLING_TEMPERATURE`**（`model_core/config.py`）：采样时使用 `logits / temperature` 再构造 Categorical；默认 `1.0`（与原行为一致）。
- **使用方式**：将 `SAMPLING_TEMPERATURE` 设为 `1.2`～`1.5` 可增强前期探索；若需随 step 衰减，可在 `engine.py` 的采样循环内根据 `step` 计算 `temperature`（例如从 1.5 线性衰减到 1.0）。
