from transformers import AutoTokenizer
from collections import defaultdict

# 示例语料库
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

# 1. 初始化BERT分词器用于预分词
print("步骤1: 使用BERT分词器进行预分词")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# 2. 统计词频
word_freqs = defaultdict(int)
for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

print("\n步骤2: 词频统计结果:")
for word, freq in word_freqs.items():
    print(f"'{word}': {freq}")

# 3. 构建字母表
alphabet = []
for word in word_freqs.keys():
    if word[0] not in alphabet:
        alphabet.append(word[0])
    for letter in word[1:]:
        if f"##{letter}" not in alphabet:
            alphabet.append(f"##{letter}")

alphabet.sort()
print("\n步骤3: 构建的字母表:")
print(alphabet)

# 4. 初始化词汇表（特殊token + 字母表）
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()
print("\n步骤4: 初始词汇表 (大小: {})".format(len(vocab)))
print(vocab)

# 5. 初始化分词（非首字符添加##前缀）
splits = {
    word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
    for word in word_freqs.keys()
}

print("\n步骤5: 初始分词示例:")
for i, (word, split) in enumerate(splits.items()):
    if i < 5:  # 只展示前5个
        print(f"'{word}': {split}")

# 6. 计算字符对分数的函数
def compute_pair_scores(splits):
    letter_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            letter_freqs[split[0]] += freq
            continue
            
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            letter_freqs[split[i]] += freq
            pair_freqs[pair] += freq
            
        letter_freqs[split[-1]] += freq

    # 计算得分公式: (pair_freq) / (freq_first * freq_second)
    scores = {
        pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }
    return scores

# 7. 合并字符对的函数
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
            
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # 处理##前缀的合并逻辑
                merge = a + b[2:] if b.startswith("##") else a + b
                split = split[:i] + [merge] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

# 8. 训练循环
vocab_size = 70
print(f"\n步骤8: 开始训练循环 (目标词汇表大小: {vocab_size})")

while len(vocab) < vocab_size:
    # 计算所有字符对的分数
    pair_scores = compute_pair_scores(splits)
    
    # 找到最高分的字符对
    best_pair = ""
    max_score = -1
    for pair, score in pair_scores.items():
        if score > max_score:
            best_pair = pair
            max_score = score
    
    # 执行合并
    splits = merge_pair(*best_pair, splits)
    
    # 生成新token
    new_token = (
        best_pair[0] + best_pair[1][2:]
        if best_pair[1].startswith("##")
        else best_pair[0] + best_pair[1]
    )
    vocab.append(new_token)
    
    # 打印当前合并信息
    print(f"合并 #{len(vocab)-len(alphabet)-5}: 合并 {best_pair} → '{new_token}' (得分: {max_score:.4f})")

# 9. 最终词汇表
print("\n步骤9: 最终词汇表 (大小: {})".format(len(vocab)))
print(vocab)

# 10. 编码函数
def encode_word(word):
    tokens = []
    while len(word) > 0:
        i = len(word)
        # 寻找最长的匹配子词
        while i > 0 and word[:i] not in vocab:
            i -= 1
        
        if i == 0:  # 未找到匹配
            return ["[UNK]"]
        
        token = word[:i]
        tokens.append(token)
        word = word[i:]
        
        # 为剩余部分添加##前缀
        if len(word) > 0:
            word = f"##{word}"
    
    return tokens

# 11. 分词函数
def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    encoded_words = [encode_word(word) for word in pre_tokenized_text]
    return sum(encoded_words, [])  # 展平列表

# 测试分词器
test_text = "This is tokenization example for Hugging Face Course!"
print("\n步骤11: 测试分词器")
print(f"原始文本: '{test_text}'")
print(f"分词结果: {tokenize(test_text)}")

# 对比测试
print("\n对比测试:")
print("'Hugging':", encode_word("Hugging"))  # 在词汇表中
print("'HOgging':", encode_word("HOgging"))  # 不在词汇表中
print("'tokenization':", encode_word("tokenization"))