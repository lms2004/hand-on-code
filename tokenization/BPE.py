import re
import collections

# --------------------------
# 工具函数：词表与词频统计
# --------------------------
def get_vocab(text):
    """ 
        1. 准备语料库 -> [词语_ 词频]基础词表
        2. 基础词表 → 拆分最小单元
    """

    # defaultdict(function_factory)
    #   构建的是一个类似dictionary的对象，
    #   1. 其中keys的值，自行确定赋值，
    #   2. 但是values的类型，是function_factory的类实例
    vocab = collections.defaultdict(int)

    # strip() 方法用于移除字符串头尾指定的字符（默认为空格）
    # split() 方法通过指定分隔符对字符串进行切片（默认为空格）
    for word in text.strip().split():
        # 输出 -> eg. 's t u d e n t s </w>'
        processed_word = ' '.join(list(word)) + ' </w>'
        vocab[processed_word] += 1
    return vocab



def get_stats(vocab):
    """统计 vocab 相邻字符对频率"""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        # 单个单词 相邻字符对频率统计
        for i in range(len(symbols)-1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

# --------------------------
# BPE合并核心操作
# --------------------------
def merge_vocab(pair, v_in):
    """合并指定的字符对并生成新词表"""
    v_out = {}

    # pair ('i', 'n') 
    # 1.    join ->  'i n'
    # 2.    re.escape(re.escape(pattern) 转义正则特殊字符) ->  'i\\ n'
    # 3.    re.compile(编译正则表达式) ->  
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        # 1. 匹配到 ->  'i n' 替换 -> 'in'
        new_word = p.sub(''.join(pair), word)
        v_out[new_word] = v_in[word]
    return v_out

# --------------------------
# 分词与词元管理
# --------------------------
def get_tokens(vocab):
    """从词表中提取词元并统计频率"""
    tokens = collections.defaultdict(int)
    for word, freq in vocab.items():
        for token in word.split():
            tokens[token] += freq
    return tokens

def get_tokens_from_vocab(vocab):
    """
        1. 生成词元频率表,用于排序 eg.  {'The</w>': 2, 'aims</w>': 1})
    
        2. 词元化映射，用于分词 eg. {'The</w>': ['The</w>'], 'aims</w>': ['aims</w>']}
    """
    tokens_freq = collections.defaultdict(int)
    tokenization_map = {}
    for word, freq in vocab.items():
        tokens = word.split()
        for token in tokens:
            tokens_freq[token] += freq
        tokenization_map[''.join(tokens)] = tokens
    return tokens_freq, tokenization_map

def measure_token_length(token):
    """计算词元长度（考虑结尾符号）"""
    return len(token[:-4]) + 1 if token.endswith('</w>') else len(token)

def sort_tokens(tokens_freq):
    """
    排序优先级
        1. 长度更长的词元会排在前面；
        2. 若长度相同，则频率更高的词元排在前面
    """
    return [token for token, _ in sorted(
        tokens_freq.items(), 
        key=lambda x: (measure_token_length(x[0]), x[1]), 
        reverse=True
    )]

def tokenize_word(word, sorted_tokens, unknown_token='</u>'):
    """
        通过贪心匹配策略
        1. 输入单词 -> 词表中存在的子词
        2. 无法匹配 -> 未知标记填充
    """
    if not word:
        return []
    if not sorted_tokens:
        return [unknown_token] * len(word)

    for i, token in enumerate(sorted_tokens):
        token_reg = re.escape(token.replace('.', '[.]'))
        matches = list(re.finditer(token_reg, word))
        if not matches:
            continue

        tokens = []
        last_pos = 0
        for match in matches:
            start, end = match.start(), match.end()
            if start > last_pos:
                tokens += tokenize_word(word[last_pos:start], sorted_tokens[i+1:], unknown_token)
            tokens.append(token)
            last_pos = end

        if last_pos < len(word):
            tokens += tokenize_word(word[last_pos:], sorted_tokens[i+1:], unknown_token)
        return tokens

    return [unknown_token] * len(word)

# --------------------------
# 主程序：BPE训练与测试
# --------------------------
if __name__ == "__main__":
    # 初始化文本和词表
    text = "   The aims for this subject is for students to develop an understanding of the main algorithms used in naturallanguage processing, for use in a diverse range of applications including text classification, machine translation, and question answering. Topics to be covered include part-of-speech tagging, n-gram language modelling, syntactic parsing and deep learning. The programming language used is Python, see for more information on its use in the workshops, assignments and installation at home."
    vocab = get_vocab(text) 
    
    # BPE训练循环
    for i in range(100):
        pairs = get_stats(vocab)
        if not pairs:
            break
        
        # 选择最高频的字符对进行合并 -> pairs = ['', ''] 词频
        best_pair = max(pairs, key=pairs.get)

        # 合并字符对并更新词表
        vocab = merge_vocab(best_pair, vocab)
        print(f"合并轮次 {i+1}: 合并字符对 {best_pair}")

    # 生成最终词元表
    tokens_freq, tokenization_map = get_tokens_from_vocab(vocab)
    sorted_tokens = sort_tokens(tokens_freq)
    print("\n最终词元表:", sorted_tokens)

    # 测试分词
    test_cases = [
        'I like natural language processing!',
        'I like natural languaaage processing!'
    ]
    
    for sentence in test_cases:
        print(f"\n测试句子: {sentence}")
        for word in sentence.split():
            processed_word = word + '</w>'

            # 单词在映射表中，直接映射
            if processed_word in tokenization_map:
                tokens = tokenization_map[processed_word]
            else:
                tokens = tokenize_word(processed_word, sorted_tokens)
            print(f"{word.ljust(15)} => {tokens}")


