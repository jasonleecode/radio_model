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
| 3b | CRNN+CTC 模型 + 训练循环 + 对照 baseline 评测 | 待办 |
| 4 | 真实电台硬化：邻台/衰落/自动增益/选频 | 待办 |

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
- `scripts/demo_synth.py` — 渲染消息到 WAV。
- `scripts/demo_dataset.py` — 检视训练样本与 batch 统计。
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

## ⚠️ 合规

接真实电台自动发射涉及无线电发射，需业余电台执照，且多数地区限制无人值守全自动发射。开发期一律用**音频回环 / 假负载 / 纯接收**测试，不真发射。
