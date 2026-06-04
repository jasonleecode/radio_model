# radio_cw — 神经网络 CW（莫尔斯电报）收发实验

用神经网络处理无线电 CW 收发：**从麦克风实时收听莫尔斯电报、解码，按内容自动应答，再把回复合成为键控音频从喇叭发出**。

探索性项目，目前已打通从合成、解码、训练到实时麦克风收发的完整链路。

```bash
pip install -r requirements.txt          # numpy scipy torch sounddevice (+ tkinter 标准库)
python scripts/live_gui.py               # 实时界面：麦克风收 CW → 解码 → 喇叭回复
```

---

## 实时 GUI

把手台/收音机的喇叭对着电脑麦克风（或线路接入），运行 `python scripts/live_gui.py`，选输入设备 → **Start**，解码报文实时滚动。

- 🎙️ **麦克风实时收 CW** → 解码 → 显示
- 🔊 **喇叭发报**：手动（输入框 + `CQ` / `599` / `73` 一键预设）或自动
- 🤖 **自动回复**：可选，**默认关闭**；开启后由 QSO 状态机自动应答
- 🔀 **解码后端界面可切**：DSP baseline / 训练的 CRNN 模型（默认 DSP，真实麦克风音频上更稳）
- 📊 实时显示 频率 / SNR / WPM / 输入电平
- **回合制**：检测到一段发报结束（静音 > 词间隔）才整段解码——CW 本就一问一答
- **半双工**：发报时自动静音麦克风，避免解码到自己的侧音

> ⚠️ **自动回复 + 接真实电台 = 自动发射**，需业余电台执照，多地限制无人值守。开发期请用音频回环 / 假负载 / 纯接收。

---

## 架构：三层松耦合

层与层之间只传 **文本 + 元数据**，每层可独立替换、独立验证。

```
 ┌──────────────┐  文本+元数据   ┌──────────────┐   文本    ┌──────────────┐
 │  感知层       │ ─────────────▶│  决策层       │ ────────▶│  执行层       │
 │ 音频→CW文本   │ {call,rst,wpm}│ 状态机/模板   │  回复串   │ 文本→音频键控 │
 └──────────────┘               └──────────────┘           └──────────────┘
        ▲ 麦克风 / 声卡 / SDR 音频流                       键控音频/侧音 │
        └─────────────────── 真实电台 / 音频回环 ────────────────────────┘
```

- **感知层** — 混合架构：DSP 旁路估 频率/SNR/WPM（元数据），CRNN+CTC 模型吃**窄带频谱图**解码文本。回合制缓冲解码（非流式）。模型与 DSP baseline 同签名、界面可切。详见 [`docs/perception_design.md`](docs/perception_design.md)。**唯一需要学习模型的一层。**
- **决策层** — 有限状态机 + 模板。呼号 / RST 等字段必须精确，故不用生成式模型。
- **执行层** — 确定性 DSP。文本→点划查表→带升降包络（防 key click）的侧音。

## 关键结果

### DSP baseline 基准（`scripts/eval_baseline.py`）

| 场景 | CER |
|------|-----|
| 机器码 @ 0 dB SNR | ~0% |
| 机器码 @ −3 dB SNR | 1–3% |
| 人手码 jitter=0.20 @ 10 dB | ~8% |
| 人手码 jitter=0.30 @ 10 dB | ~30–36% |

白噪声下机器码 baseline 已近乎完美，**模型的价值集中在人手 fist 节奏**。

### 模型 vs baseline（CRNN 训练 12000 步，CER %，DSP / 模型；`scripts/eval_model.py`）

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

**诚实结论**：干净机器码上 DSP 略胜（0% vs 模型 ~1% 残差地板）；但人手节奏抖动越大，模型优势越明显（j=0.3 时约 2× 更好），极低 SNR 也略占优。模型恰好在它被设计来对付的"凌乱人手信号"上赢——验证了项目核心假设。

> 已知局限：(1) 闭环 QSO 是"全有或全无"测试（需连续 ~6 次准确解呼号），不奖励模型的 CER 优势，低抖动时 DSP 的精确性反而占优；(2) 模型训练时 SNR ∈ [−3,25]dB，**没见过无噪声音频**，纯净信道会解码失败（OOD），真实电台/麦克风从无纯净信道故影响不大。待改进：训练增强纳入近无噪声样本。

## 命令速查

```bash
# 实时收发界面
python scripts/live_gui.py

# 执行层：渲染一条消息到 WAV
python scripts/demo_synth.py "CQ CQ DE BG1ABC K" --wpm 20 -o cq.wav

# baseline 基准 / 端到端闭环
python scripts/eval_baseline.py
python scripts/demo_qso.py --snr 5                       # DSP 驱动
python scripts/demo_qso.py --model runs/crnn.pt          # 模型驱动 (需 GPU 环境)

# 训练（见下方"训练环境"）
python scripts/train_model.py --smoke                    # CPU 自检：过拟合一小批
<cuda-torch>/bin/python scripts/train_model.py --steps 12000 --batch-size 32 \
    --device cuda --save runs/crnn.pt
<cuda-torch>/bin/python scripts/eval_model.py runs/crnn.pt --n 40

# 测试
python -m pytest tests/ -q
```

## 代码结构

**执行层**
- `radio_cw/morse.py` — 国际莫尔斯表 + prosign（`<AR>` `<SK>` …），文本↔点划互转。
- `radio_cw/synth.py` — 合成器：文本→键控侧音 + 元素时间线 `Timeline`（训练标签来源）；`timing_jitter`/`weight_bias` 模拟人手 fist；`estimate_duration` 不渲染估时长。

**感知层**
- `radio_cw/dsp_frontend.py` — 锁频 + Hilbert 包络 + 元数据(freq/SNR/WPM) 旁路。
- `radio_cw/decode_dsp.py` — DSP baseline 解码器（自适应阈值 + 时长聚类 + 查表）。
- `radio_cw/features.py` — 窄带 log-频谱图（短窗短跳步，16 bins / 250 Hz 帧率）。
- `radio_cw/vocab.py` — CTC 词表/tokenizer，莫尔斯码冲突归一（`=`/`<BT>` 等）。
- `radio_cw/model.py` — CRNN+CTC 模型（~0.5M 参数，CNN→BiGRU→Linear）。
- `radio_cw/perception.py` — 统一感知层：`DSPPerception` / `ModelPerception` 同签名互换，元数据均走 DSP 旁路。

**决策层**
- `radio_cw/decision.py` — QSO 状态机 + 模板，SNR→RST，来报解析。

**训练 / 数据**
- `radio_cw/corpus.py` — 拟真 CW 语料（呼号/RST/话术/缩写）。
- `radio_cw/channel.py` — 信道损伤（白噪声/QSB/QRM），eval 与训练增强共用。
- `radio_cw/dataset.py` — 在线合成数据集 + 6 维增强 + CTC collate + 时长截断。
- `radio_cw/train.py` — CTC 训练循环 + 贪心解码 CER 监控 + checkpoint。
- `radio_cw/metrics.py` — 编辑距离 / CER。

**实时**
- `radio_cw/audio_io.py` — 声卡设备枚举 / 8k 重采样 / 播放。
- `radio_cw/live.py` — `CWActivityGate`（流式分段）+ `LiveEngine`（麦克风→解码→可选自动回复→喇叭，半双工）。

**脚本**：`demo_synth` · `demo_dataset` · `demo_qso` · `eval_baseline` · `eval_model` · `train_model` · `live_gui`

## 路线图

| 阶段 | 内容 | 状态 |
|------|------|------|
| 0 | 执行层：文本→键控音频（兼训练数据合成器） | ✅ |
| 1 | DSP baseline 解码器（闭环 + 对照基线 + 元数据旁路） | ✅ |
| 2 | 决策层状态机 + 端到端闭环 | ✅ |
| 3a | 数据管线：词表/语料/特征/数据集 + 6 维增强 | ✅ |
| 3b | CRNN+CTC 模型 + 训练循环（smoke test） | ✅ |
| 3c | 正式训练 + 对照 baseline 评测 | ✅ |
| 3d | 模型接进闭环（统一感知层，模型/DSP 可切） | ✅ |
| 4a | 实时 GUI：麦克风收 CW → 解码 → 喇叭回复 | ✅ |
| 4b | 真实电台硬化：邻台/衰落/AGC/选频；真实录音微调模型 | 待办 |

## 训练环境

- **实时 GUI / 推理 / 测试**：用带 `tkinter` 的 CPU 环境即可（如 Python 3.13 + `torch` CPU 版）。单段报文推理 CPU 足够。
- **正式训练**：需要一个 **CUDA 可用的 torch 环境**（本项目在 RTX 3080 + `torch 2.x+cu121` 上训练，12000 步约几十分钟）。`--smoke` 自检在 CPU 上即可跑。
- 注意：模型权重（`runs/*.pt`）和 `*.wav` 不入库（见 `.gitignore`）。

## ⚠️ 合规

接真实电台自动发射涉及无线电发射，需业余电台执照，且多数地区限制无人值守全自动发射。开发期一律用**音频回环 / 假负载 / 纯接收**测试，不真发射。
