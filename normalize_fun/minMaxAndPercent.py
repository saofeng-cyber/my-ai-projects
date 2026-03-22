import math


def robust_min_max_normalize(scores, epsilon=1e-6):
    """
    稳健的 Min-Max 归一化
    :param scores: 原始分数列表 (List[float])
    :param epsilon: 极小值，防止分母为0
    :return: 归一化后的分数列表 (List[float])，范围 [0, 1]
    """
    if not scores:
        return []

    min_val = min(scores)
    max_val = max(scores)

    # 1. 处理所有分数相同的情况 (分母为0)
    if math.isclose(max_val, min_val, abs_tol=epsilon):
        # 如果所有分数都一样，全部设为 0.5 (或者 1.0，视策略而定)
        # 这里设为 0.5 表示“中等置信度”，避免极端值
        return [0.5] * len(scores)

    range_val = max_val - min_val

    normalized = []
    for s in scores:
        # 2. 核心公式
        norm_s = (s - min_val) / range_val
        normalized.append(norm_s)
    return normalized


def hybrid_search_with_min_max(bm25_docs, vector_docs, alpha=0.5):
    """
    基于 Min-Max 归一化的混合检索融合
    :param bm25_docs: [(doc_id, score), ...] 按分数降序
    :param vector_docs: [(doc_id, score), ...] 按分数降序
    :param alpha: 向量检索的权重 (0~1)，(1-alpha) 为 BM25 权重
    :return: 融合后的排序结果 [(doc_id, final_score), ...]
    """
    # 1. 提取纯分数列表
    bm25_scores = [s for _, s in bm25_docs]
    vector_scores = [s for _, s in vector_docs]

    # 2. 分别进行归一化
    norm_bm25_scores = robust_min_max_normalize(bm25_scores)
    norm_vector_scores = robust_min_max_normalize(vector_scores)

    print("norm_bm25_scores", norm_bm25_scores)
    print("norm_vector_scores", norm_vector_scores)

    # 3. 构建 ID -> 分数的映射字典
    # 注意：未出现在某个列表中的文档，该部分得分为 0
    score_map = {}

    # 处理 BM25 结果
    for i, (doc_id, _) in enumerate(bm25_docs):
        if doc_id not in score_map:
            score_map[doc_id] = {'bm25': 0, 'vec': 0}
        score_map[doc_id]['bm25'] = norm_bm25_scores[i]

    # 处理 向量 结果
    for i, (doc_id, _) in enumerate(vector_docs):
        if doc_id not in score_map:
            score_map[doc_id] = {'bm25': 0, 'vec': 0}
        score_map[doc_id]['vec'] = norm_vector_scores[i]

    print("score_map", score_map)
    # 4. 加权融合
    final_results = []
    for doc_id, scores in score_map.items():
        # 公式: Final = alpha * Vector + (1-alpha) * BM25
        final_score = (alpha * scores['vec']) + ((1 - alpha) * scores['bm25'])
        final_results.append((doc_id, final_score))

    # 5. 按最终分数降序排序
    final_results.sort(key=lambda x: x[1], reverse=True)

    return final_results

def percentile_normalize(docs_with_scores):
    """
    百分位归一化
    :param docs_with_scores: [(doc_id, score), ...] 列表
    :return: [(doc_id, normalized_score), ...] 分数范围 [0, 1]
    """
    if not docs_with_scores:
        return []

    n = len(docs_with_scores)
    # 按分数降序排序，确保排名正确
    # 注意：如果有相同分数，它们将获得相同的百分位（取决于具体策略，这里采用平均排名或最大排名）
    # 为了简单和稳健，我们直接使用排名计算：(N - rank) / N

    # 先排序
    sorted_docs = sorted(docs_with_scores, key=lambda x: x[1], reverse=True)

    normalized_results = []

    for i, (doc_id, score) in enumerate(sorted_docs):
        # 策略 A: 简单线性映射
        # 第 1 名 (i=0) -> 1.0
        # 最后一名 (i=n-1) -> 1/n (接近 0)
        # 公式: (n - i) / n
        norm_score = (n - i) / n

        # 策略 B (可选): 映射到 [0, 1] 且最后一名为 0
        # norm_score = (n - 1 - i) / (n - 1) if n > 1 else 1.0

        normalized_results.append((doc_id, norm_score))

    return normalized_results


def hybrid_search_percentile(bm25_docs, vector_docs, alpha=0.5):
    """
    基于百分位归一化的混合检索
    """
    # 1. 分别归一化
    norm_bm25 = percentile_normalize(bm25_docs)
    norm_vector = percentile_normalize(vector_docs)

    print("norm_bm25", norm_bm25)
    print("norm_vector", norm_vector)

    # 2. 构建映射
    score_map = {}

    # 填充 BM25
    for doc_id, score in norm_bm25:
        if doc_id not in score_map:
            score_map[doc_id] = {'bm25': 0, 'vec': 0}
        score_map[doc_id]['bm25'] = score

    # 填充 Vector
    for doc_id, score in norm_vector:
        if doc_id not in score_map:
            score_map[doc_id] = {'bm25': 0, 'vec': 0}
        score_map[doc_id]['vec'] = score

    # 3. 加权融合
    final_results = []
    for doc_id, scores in score_map.items():
        final_score = (alpha * scores['vec']) + ((1 - alpha) * scores['bm25'])
        final_results.append((doc_id, final_score))

    # 4. 排序
    final_results.sort(key=lambda x: x[1], reverse=True)
    return final_results


# --- 模拟极端场景测试 ---

# 场景：BM25 出现超级离群值 (Doc_A 100分，其他都在 2-5分)
bm25_extreme = [
    ('Doc_A', 100.0),
    ('Doc_B', 4.5),
    ('Doc_C', 4.2),
    ('Doc_D', 2.1),
    ('Doc_E', 2.0)
]

# 场景：向量检索认为 Doc_B 语义最相关 (排第一)
vector_semantic = [
    ('Doc_B', 0.95),
    ('Doc_C', 0.92),
    ('Doc_A', 0.60),  # Doc_A 语义其实不太相关
    ('Doc_D', 0.55),
    ('Doc_E', 0.50)
]

print("🧪 场景测试：离群值对抗")
results_minmax = hybrid_search_with_min_max(bm25_extreme, vector_semantic, alpha=0.5)  # 复用之前的 Min-Max 函数
results_pct = hybrid_search_percentile(bm25_extreme, vector_semantic, alpha=0.5)
print(results_minmax)
print("\n")
print(results_pct)
