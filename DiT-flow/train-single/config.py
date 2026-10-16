# 实际使用 min(SUBSET_SIZE, 所选作者字数)；至少需要 max_references + 1 个字。
SUBSET_SIZE = 16

# 是否加入风格提取隐状态的作者损失；只影响 train-single，不影响 train-full。
ENABLE_STYLE_HIDDEN_LOSS = True
