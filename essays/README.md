# 语音克隆论文

下载日期：2026-10-03。PDF 来自 arXiv；文件名前的年月为论文首次公开时间。下载使用不带版本号的链接，获取下载时 arXiv 提供的版本。

这些论文用于比较架构，不构成当前效果排名。论文中的结果仅适用于其报告的模型版本、测试集和指标。

| 建议阅读顺序 | 论文 | 本地 PDF | 原文 | 阅读重点 |
|---|---|---|---|---|
| 1 | Qwen3-TTS Technical Report | [PDF](2026-01_Qwen3-TTS_2601.15621.pdf) | [arXiv:2601.15621](https://arxiv.org/abs/2601.15621) | 比较说话人特征表示与参考文本—语音对两种克隆方式。 |
| 2 | MiniMax-Speech: Intrinsic Zero-Shot Text-to-Speech with a Learnable Speaker Encoder | [PDF](2025-05_MiniMax-Speech_2505.07916.pdf) | [arXiv:2505.07916](https://arxiv.org/abs/2505.07916) | 研究说话人编码器如何提取音色，以及如何与生成模型一起训练。 |
| 3 | VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation and True-to-Life Voice Cloning | [PDF](2025-09_VoxCPM_2509.24650.pdf) | [arXiv:2509.24650](https://arxiv.org/abs/2509.24650) | 研究不依赖外部离散语音编码器的生成方法，以及内容和声音细节的分层表示。 |
| 4 | Fish Audio S2 Technical Report | [PDF](2026-03_Fish-Audio-S2_2603.08823.pdf) | [arXiv:2603.08823](https://arxiv.org/abs/2603.08823) | 研究参考音频的输入方式、主要表示与细节的分别生成，以及多个质量目标的联合训练。 |
| 5 | MOSS-TTS Technical Report | [PDF](2026-03_MOSS-TTS_2603.18090.pdf) | [arXiv:2603.18090](https://arxiv.org/abs/2603.18090) | 比较两种生成结构，重点看音频帧内部的局部生成模块。 |
| 6 | CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training | [PDF](2025-05_CosyVoice-3_2505.17589.pdf) | [arXiv:2505.17589](https://arxiv.org/abs/2505.17589) | 研究编码器训练任务、数据规模和后续训练对生成效果的影响。 |

迁移到字图任务时，需要另外保证：任意调换参考对的排列，不改变输出；每对样本字与楷体字的对应关系、每张字图内部的二维位置均保留。上述语音论文不直接证明这一性质。
