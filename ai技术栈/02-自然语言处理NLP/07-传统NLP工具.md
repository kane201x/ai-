# 传统 NLP 工具

## 1. 中文分词

### 主流分词工具
| 工具 | 特点 | 速度 | 准确率（PKU） | 支持自定义词典 |
|------|------|------|-------------|--------------|
| jieba | 最流行，精确/全/搜索引擎模式 | 快 | 85.4% | 是 |
| HanLP | 功能全面，支持多任务 | 中 | 93.2% | 是 |
| LTP（哈工大） | 成熟，支持多种语言分析 | 中 | 94.2% | 是 |
| Pkuseg | 北大研发，多领域模型 | 中 | 95.3% | 是 |
| THULAC | 清华研发，准确率高 | 快 | 94.6% | 否 |

### 词典 vs 统计
- **词典分词**：最大正向/逆向匹配
- **统计分词**：HMM/CRF/BI-LSTM-CRF

### jieba 分词核心实现

```python
class JiebaTokenizer:
    def __init__(self, dict_path=None):
        self.word_freq = {}
        self.total = 0
        self.max_word_len = 0
        if dict_path:
            self.load_dict(dict_path)

    def load_dict(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word, freq = parts[0], int(parts[1])
                    self.word_freq[word] = freq
                    self.total += freq
                    self.max_word_len = max(self.max_word_len, len(word))

    def dag(self, sentence):
        n = len(sentence)
        dag = {}
        for i in range(n):
            dag[i] = []
            for j in range(i + 1, min(i + self.max_word_len + 1, n + 1)):
                word = sentence[i:j]
                if word in self.word_freq or len(word) == 1:
                    dag[i].append(j)
        return dag

    def cut(self, sentence):
        dag = self.dag(sentence)
        route = {}
        n = len(sentence)
        route[n] = (0, 0)
        for i in range(n - 1, -1, -1):
            route[i] = max((self.word_freq.get(sentence[i:x], 1) / self.total + route[x][0], x) for x in dag[i])
        words = []
        i = 0
        while i < n:
            _, next_pos = route[i]
            words.append(sentence[i:next_pos])
            i = next_pos
        return words
```

### 分词算法对比
| 算法 | 原理 | 优点 | 缺点 | 准确率 |
|------|------|------|------|-------|
| 正向最大匹配 | 从左向右匹配最长词 | 简单、快速 | 无法消歧 | 80% |
| 逆向最大匹配 | 从右向左匹配最长词 | 比正向略优 | 同上 | 81% |
| 双向最大匹配 | 正反比较选择 | 减少错误 | 仍有限 | 83% |
| HMM | 隐马尔可夫模型 | 处理未登录词 | 训练复杂 | 88% |
| CRF | 条件随机场 | 特征灵活 | 特征工程复杂 | 92% |
| BiLSTM-CRF | 深度序列标注 | 自动化特征 | 需标注数据 | 95% |

## 2. 词向量

### Word2Vec（2013）
- **CBOW**：上下文预测当前词
- **Skip-gram**：当前词预测上下文
- **训练技巧**：负采样、层次 Softmax

### GloVe（2014）
- **全局矩阵分解**：共现矩阵 + 词向量
- **优势**：利用全局统计信息

### FastText（2016）
- **子词 N-gram**：捕捉形态学信息
- **OOV 友好**：未登录词也可生成向量

### TF-IDF 实现

```python
import math
from collections import Counter

class TFIDF:
    def __init__(self):
        self.doc_freq = Counter()
        self.doc_count = 0

    def fit(self, documents):
        self.doc_count = len(documents)
        for doc in documents:
            terms = set(doc)
            for term in terms:
                self.doc_freq[term] += 1

    def transform(self, tokens):
        tf = Counter(tokens)
        total = len(tokens)
        vec = {}
        for term, count in tf.items():
            tf_val = count / total
            idf_val = math.log((self.doc_count + 1) / (self.doc_freq.get(term, 0) + 1)) + 1
            vec[term] = tf_val * idf_val
        return vec

    def similarity(self, vec1, vec2):
        common = set(vec1.keys()) & set(vec2.keys())
        dot = sum(vec1[t] * vec2[t] for t in common)
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        return dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0
```

### N-gram 语言模型实现

```python
from collections import defaultdict

class NgramLM:
    def __init__(self, n=3, smoothing="kneser_ney"):
        self.n = n
        self.smoothing = smoothing
        self.counts = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)

    def fit(self, corpus):
        for sentence in corpus:
            tokens = ["<BOS>"] * (self.n - 1) + sentence + ["<EOS>"]
            for i in range(self.n - 1, len(tokens)):
                context = tuple(tokens[i - self.n + 1:i])
                word = tokens[i]
                self.counts[context][word] += 1
                self.context_counts[context] += 1

    def probability(self, word, context):
        context_count = self.context_counts.get(context, 0)
        word_count = self.counts[context].get(word, 0)
        if self.smoothing == "add_one":
            return (word_count + 1) / (context_count + len(self.counts[context]))
        elif self.smoothing == "kneser_ney":
            d = 0.75
            if context_count == 0:
                return self.backoff_probability(word, context[1:])
            discount = max(word_count - d, 0) / context_count
            if discount > 0:
                return discount
            else:
                return d * self.continuation_probability(word, context) / context_count
        else:
            return word_count / context_count if context_count > 0 else 0

    def continuation_probability(self, word, context):
        unique_contexts = sum(1 for ctx, cnt in self.counts.items() if word in cnt)
        total_contexts = sum(len(ctx) for ctx in self.counts.values())
        return unique_contexts / total_contexts if total_contexts > 0 else 0
```

### 词向量工具对比
| 工具 | 原理 | 上下文 | OOV处理 | 训练速度 | 适用 |
|------|------|--------|--------|---------|------|
| Word2Vec | 预测式 | 局部窗口 | 无 | 快 | 通用嵌入 |
| GloVe | 矩阵分解 | 全局共现 | 无 | 中 | 中等语料 |
| FastText | 子词+N-gram | 局部窗口 | 好 | 快 | 形态学丰富语言 |
| LSA/SVD | 矩阵分解 | 文档级 | 无 | 慢 | 主题分析 |

## 3. 传统语言模型

### N-gram LM
- **统计**：P(w_n|w_{n-1},...,w_{n-N+1})
- **平滑**：Kneser-Ney / Good-Turing

| 平滑方法 | 原理 | 效果 | 复杂度 |
|---------|------|------|-------|
| 加一平滑 | 每个词+1 | 差 | 低 |
| Good-Turing | 基于频次估计 | 中 | 中 |
| Kneser-Ney | 连续概率+折扣 | 最好 | 高 |
| Witten-Bell | 基于计数折扣 | 较好 | 中 |

## 4. 句法分析工具
- **Stanford CoreNLP**：多语言句法分析
- **SyntaxNet / Parsey McParseface**：Google 依存分析
- **spaCy**：工业级 NLP 工具

### 工具对比
| 工具 | 语言 | 句法类型 | 速度 | 准确率 | 特点 |
|------|------|---------|------|-------|------|
| Stanford CoreNLP | 多语言 | 成分+依存 | 慢 | 高 | 功能最全 |
| spaCy | 多语言 | 依存 | 极快 | 高 | 工业级标准 |
| SyntaxNet | 多语言 | 依存 | 慢 | 很高 | Google 出品 |
| Stanza | 多语言 | 依存 | 中 | 很高 | CoreNLP 继任 |

## 5. 信息提取
- **正则表达式**：模式匹配
- **规则系统**：基于模式的关系抽取
- **CRF**：序列标注式信息提取

### 规则关系抽取示例

```python
class RuleBasedRE:
    def __init__(self):
        self.patterns = [
            ("生于", "出生地", r"(\S+)\s*生于\s*(\S+)"),
            ("位于", "位置", r"(\S+)\s*位于\s*(\S+)"),
            ("成立于", "成立时间", r"(\S+)\s*成立于\s*(\S+)"),
            ("担任", "职位", r"(\S+)\s*担任\s*(\S+)"),
        ]

    def extract(self, text, pattern_name=None):
        results = []
        patterns = [p for p in self.patterns if pattern_name is None or p[0] == pattern_name]
        for rel_name, rel_type, pattern in patterns:
            import re
            for m in re.finditer(pattern, text):
                results.append({
                    "relation": rel_type,
                    "subject": m.group(1),
                    "object": m.group(2),
                    "trigger": rel_name,
                })
        return results
```

## 6. 现代定位
- 大模型时代传统工具多被 LLM 替代
- 但仍用于：预处理、轻量任务、资源受限场景

### 传统 vs 深度 vs LLM 对比
| 任务 | 传统方法 | 深度方法 | LLM 方法 |
|------|---------|---------|---------|
| 分词 | jieba 词典（0.1ms） | BiLSTM-CRF（10ms） | Prompt 分词（500ms） |
| 词性标注 | HMM（0.5ms） | BERT-Tagger（15ms） | In-context（1s） |
| NER | CRF + 特征（2ms） | BERT-CRF（30ms） | In-context（1s） |
| 文本分类 | TF-IDF + SVM（1ms） | BERT 微调（20ms） | LLM Prompt（500ms） |
| 依存分析 | 图算法（5ms） | Biaffine（25ms） | LLM 推理（1s+） |

### 工具栈选择指南
| 场景 | 推荐工具 |
|------|---------|
| 快速原型、轻量任务 | jieba + TF-IDF + LR |
| 学术研究、高精度 | HanLP / LTP + BERT |
| 工业级生产、低延迟 | spaCy + FastText |
| 复杂理解、多语言 | LLM API + RAG |
| 资源受限、嵌入式 | jieba + N-gram + 规则 |
