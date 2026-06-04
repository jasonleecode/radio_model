# radio_cw — 神经网络 CW（莫尔斯电报）收发实验

探索用神经网络处理无线电 CW 收发：从音频解码莫尔斯电报，按内容自动应答，再把回复合成为键控音频。

## 架构：三层松耦合

层与层之间只传 **文本 + 元数据**，每层可独立替换、独立验证。

```
 ┌──────────────┐  文本+元数据   ┌──────────────┐   文本    ┌──────────────┐
 │  感知层       │ ─────────────▶│  决策层       │ ────────▶│  执行层       │
 │ 音频→CW文本   │ {call,rst,wpm}│ 状态机/模板   │  回复串   │ 文本→音频键控 │
 └──────────────┘               └──────────────┘           └──────────────┘
        ▲ 声卡/SDR 音频流                                  键控音频/侧音 │
        └─────────────────── 真实电台 / 音频回环 ────────────────────────┘
```

- **感知层** — 混合架构：DSP 旁路估频率/SNR/WPM（元数据），CRNN+CTC 模型吃**窄带频谱图**解码文本。回合制缓冲解码（非流式）。详见 [`docs/perception_design.md`](docs/perception_design.md)。**唯一需要学习模型的一层。**
- **决策层** — 有限状态机 + 模板。呼号 / RST 等字段必须精确，故不用生成式模型。
- **执行层** — 确定性 DSP。文本→点划查表→带升降包络的侧音。✅ 已完成。

## 路线图

| 阶段 | 内容 | 状态 |
|------|------|------|
| 0 | 执行层：文本→键控音频（兼作训练数据合成器） | ✅ 完成 |
| 1 | DSP baseline 解码器（建立闭环 + 模型对照基线，**复用为元数据旁路**） | ✅ 完成 |
| 2 | 决策层状态机（最小自动应答 demo 成立） | ✅ 完成 |
| 3a | 数据管线：词表/语料/特征/数据集 + 6 维增强 | ✅ 完成 |
| 3b | CRNN+CTC 模型 + 训练循环（smoke test 通过） | ✅ 完成 |
| 3c | 正式训练 + 对照 baseline 评测（CER 分桶） | ✅ 完成 |
| 3d | 模型接进闭环（统一感知层，模型/DSP 可互换） | ✅ 完成 |
| 4a | 实时 GUI：麦克风收 CW → 解码 → 喇叭回复（自动回复可选） | ✅ 完成 |
| 4b | 真实电台硬化：邻台/衰落/自动增益/选频 | 待办 |

## 已实现

- `radio_cw/morse.py` — 国际莫尔斯表 + 常用 prosign（`<AR>` `<SK>` …），文本↔点划互转。
- `radio_cw/synth.py` — 执行层合成器。返回音频 **和** 元素时间线 `Timeline`（训练标签来源）。支持 `timing_jitter`/`weight_bias` 模拟人手 fist。
- `radio_cw/dsp_frontend.py` — 锁频 + Hilbert 包络 + 元数据(freq/SNR/WPM) 旁路。
- `radio_cw/decode_dsp.py` — baseline 解码器（自适应阈值 + 时长聚类 + 查表）。
- `radio_cw/channel.py` — 信道损伤（白噪声/QSB/QRM），eval 与 stage-3 增强共用。
- `radio_cw/metrics.py` — 编辑距离 / CER。
- `radio_cw/decision.py` — 决策层 QSO 状态机 + 模板，SNR→RST，来报解析。
- `radio_cw/vocab.py` — CTC 词表/tokenizer，处理莫尔斯码冲突归一（`=`/`<BT>` 等）。
- `radio_cw/corpus.py` — 拟真 CW 语料生成器（呼号/RST/话术/缩写）。
- `radio_cw/features.py` — 窄带 log-频谱图（短窗短跳步，16 bins / 250 Hz）。
- `radio_cw/dataset.py` — 在线合成数据集 + 6 维增强 + CTC collate + 时长截断。
- `radio_cw/model.py` — CRNN+CTC 模型（~0.5M 参数，CNN→BiGRU→Linear）。
- `radio_cw/train.py` — CTC 训练循环 + 贪心解码 CER 监控。
- `radio_cw/perception.py` — 统一感知层：`DSPPerception` / `ModelPerception` 同签名互换，元数据均来自 DSP 旁路（混合架构）。
- `scripts/demo_synth.py` — 渲染消息到 WAV。
- `scripts/demo_dataset.py` — 检视训练样本与 batch 统计。
- `scripts/train_model.py` — 训练模型；`--smoke` 跑过拟合自检。
- `scripts/eval_model.py` — 模型 vs DSP baseline 同数据对照评测（CER 分桶）。
- `radio_cw/audio_io.py` — 声卡设备枚举 / 8k 重采样 / 播放。
- `radio_cw/live.py` — 实时引擎：CW 活动门（流式分段）+ LiveEngine（麦克风→解码→可选自动回复→喇叭，半双工）。
- `scripts/live_gui.py` — Tkinter 实时界面。

### 实时 GUI（阶段 4a）

把手台/收音机的喇叭对着电脑麦克风（或线路接入），运行：

```bash
python scripts/live_gui.py
```

选输入设备 → Start，解码的报文实时滚动显示。功能：
- **解码后端界面可切**：DSP baseline / 训练的模型（默认 DSP，真实麦克风音频上更稳）。
- **回合制**：检测到一段发报结束（静音 > 词间隔）才整段解码——CW 本就一问一答。
- **半双工**：发报时自动静音麦克风，避免解码到自己的侧音。
- **自动回复（默认关闭）**：开启后由 QSO 状态机自动应答；关闭时用输入框手动发，或 CQ / 599 / 73 一键预设。
- 实时显示检测到的 频率 / SNR / WPM / 输入电平。

> ⚠️ 自动回复 + 接真实电台 = 自动发射，需执照且多地限制无人值守。开发期建议音频回环/假负载/纯接收。
- `scripts/eval_baseline.py` — baseline CER 基准（机器码 vs 人手 fist）。
- `scripts/demo_qso.py` — 端到端闭环：两台站穿过音频管线完成一次完整通联。

```bash
python scripts/demo_synth.py "CQ CQ DE BG1ABC K" --wpm 20 -o cq.wav
python scripts/eval_baseline.py            # 打印 baseline 对照基准
python scripts/demo_qso.py --snr 5         # 跑完整 QSO 闭环
python -m pytest tests/ -q
```

### baseline 基准（阶段 3 模型要超越的对照组）

| 场景 | CER |
|------|-----|
| 机器码 @ 0 dB SNR | ~0% |
| 机器码 @ −3 dB SNR | 1–3% |
| 人手码 jitter=0.20 @ 10 dB | ~8% |
| 人手码 jitter=0.30 @ 10 dB | ~30–36% |

结论：白噪声下机器码 baseline 已近乎完美，**模型的价值集中在人手 fist 节奏**。

### 模型 vs baseline 对照（CRNN 训练 12000 步后，CER %，DSP / 模型）

```bash
torch_env/bin/python scripts/eval_model.py runs/crnn.pt --n 40
```

机器码（DSP 在干净时序上精确）：

| WPM | 20dB | 0dB | −3dB |
|-----|------|-----|------|
| 20  | 0.0 / 1.0 | 0.0 / 1.1 | 1.6 / 1.1 |

人手 fist @ 10dB（决定性对照）：

| WPM | j=0.0 | j=0.20 | j=0.30 |
|-----|-------|--------|--------|
| 15  | 0.0 / 1.1 | 15.7 / **4.0** | 34.8 / **14.9** |
| 20  | 0.0 / 1.1 | 6.6 / **3.7**  | 32.5 / **15.5** |
| 30  | 0.0 / 1.0 | 6.9 / **2.4**  | 28.7 / **15.1** |

**诚实结论**：干净机器码上 DSP 略胜（0% vs 模型 ~1% 残差地板）；但人手节奏抖动越大，模型优势越明显（j=0.3 时约 2× 更好），极低 SNR 也略占优。模型恰好在它被设计来对付的"凌乱人手信号"上赢。验证了项目核心假设。

### 模型接进闭环（阶段 3d）

```bash
torch_env/bin/python scripts/demo_qso.py --model runs/crnn.pt   # 模型驱动整个 QSO
python scripts/demo_qso.py --jitter 0.3                          # DSP，对比
```

模型成功接入闭环，完成完整 QSO（最初设想的"感知层用模型"实现了）。两个诚实发现：

1. **闭环是"全有或全无"的严苛测试**：一次通联要连续 ~6 次准确解出呼号才不中断。模型的 CER 优势体现在 `eval_model.py` 的分桶表里，而非二元的 QSO 成功率——低抖动时 DSP 的"精确性"反而占优，高抖动时两者在严苛解析下都吃力。
2. **噪声分布外（OOD）bug**：模型训练时 SNR ∈ [−3,25]dB，**从没见过无噪声音频**，所以纯净信道解码会崩（per-clip 归一化把静音压到 log 地板，分布外）。真实电台从无纯净信道，故 demo 的模型路径默认加轻噪声。**待改进**：训练增强应纳入偶发的极高 SNR/近无噪声样本。

## ⚠️ 训练环境注意（GPU）

当前装的是 `torch 2.11+cu130`，但本机 NVIDIA 驱动是 CUDA 12.8，版本太旧，`torch.cuda.is_available()` 为 False，只能 CPU 训练。正式训练（阶段 3c）前需二选一：
- 装一个匹配驱动的 torch（如 `cu121`/`cu128` 构建），或
- 升级 NVIDIA 驱动到支持 CUDA 13。

smoke test 在 CPU 上即可跑（~500 步几分钟）。

## ⚠️ 合规

接真实电台自动发射涉及无线电发射，需业余电台执照，且多数地区限制无人值守全自动发射。开发期一律用**音频回环 / 假负载 / 纯接收**测试，不真发射。
