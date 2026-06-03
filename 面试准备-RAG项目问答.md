# RAG 知识库问答系统 — 面试准备文档

> 基于项目 `nn_code/rag_dataset` 的实习面试模拟问答，涵盖项目架构、核心技术、设计决策、工程实践等方面。

---

## 目录

1. [项目概述类](#1-项目概述类)
2. [RAG 架构与原理类](#2-rag-架构与原理类)
3. [LangGraph 工作流类](#3-langgraph-工作流类)
4. [向量数据库与检索类](#4-向量数据库与检索类)
5. [多路召回与融合排序类](#5-多路召回与融合排序类)
6. [Embedding 与模型类](#6-embedding-与模型类)
7. [工程实践与架构类](#7-工程实践与架构类)
8. [系统设计与扩展类](#8-系统设计与扩展类)
9. [场景题 / 故障排查类](#9-场景题--故障排查类)
10. [行为与软技能类](#10-行为与软技能类)

---

## 1. 项目概述类

### Q1：请简单介绍一下这个项目是做什么的？

**理想答案：**

这是一个面向企业知识库的 **RAG（检索增强生成）问答系统**，核心功能是将产品手册、操作说明书等 PDF/Markdown 文档导入到向量数据库，然后用户可以用自然语言提问，系统通过检索相关文档片段并结合大模型生成准确答案。

系统分为两大核心流程：

- **知识导入流程（Import Pipeline）**：上传文件 → PDF 转 Markdown → 图片处理 → 文档智能切分 → 项目名识别 → BGE-M3 向量化 → 存入 Milvus 向量数据库。
- **智能问答流程（Query Pipeline）**：用户提问 → 实体识别 + 问题重写 → 三路并行召回（向量检索 + HyDE 假设性答案检索 + MCP 网络搜索）→ RRF 融合排序 → BGE-Reranker 精排 → 大模型生成答案 → SSE 流式返回前端。

技术栈：**LangGraph + FastAPI + Milvus + BGE-M3 + MongoDB + MinIO + MCP 协议**，整个工作流用 LangGraph 的状态图编排，支持流式与非流式两种响应模式。

---

### Q2：这个系统解决了什么实际业务问题？

**理想答案：**

1. **知识检索效率低**：传统方式需要人工翻阅产品操作手册、维修指南等大量文档，本系统支持自然语言直接提问，秒级获取答案。
2. **检索准确性问题**：单纯的向量检索容易漏召回或误召回，系统通过三路并行召回 + RRF 融合 + Reranker 精排 + 断崖截断算法，大幅提升 Top-K 文档的精准度。
3. **口语化提问理解差**：用户口语化提问（如"这玩意儿咋修"），系统通过 LLM 自动完成问题重写和实体识别，将模糊问题规范化。
4. **多源信息整合**：本地知识库可能不完整，系统通过 MCP 协议接入网络搜索，补充外部知识。

---

## 2. RAG 架构与原理类

### Q3：说说 RAG 的基本原理，以及你项目中 RAG 的完整流程是怎样的？

**理想答案：**

**RAG（Retrieval-Augmented Generation）** 的核心思想是：在 LLM 生成答案之前，先从外部知识库检索相关文档片段，将检索结果作为上下文注入到 Prompt 中，让 LLM 基于这些"参考资料"生成答案，从而有效缓解大模型的幻觉问题（Hallucination）和知识滞后问题。

**我项目的完整 RAG 流程：**

```
用户提问 → 实体识别 + 问题重写（LLM）
         ↓
    三路并行召回：
    ├── 稠密+稀疏向量混合检索（BGE-M3 → Milvus）
    ├── HyDE 假设性答案检索（LLM 生成假设答案 → 向量检索）
    └── MCP 网络搜索（外部知识补充）
         ↓
    RRF 倒数排名融合（加权融合三路结果）
         ↓
    BGE-Reranker 精排（Cross-encoder 逐对打分）
         ↓
    断崖截断算法（Top-K 自适应截断）
         ↓
    组装 Prompt → LLM 生成答案 → SSE 流式返回
```

**亮点：**
- 不只是简单的"问题→检索→生成"，而是在检索前增加了**问题重写**和**HyDE 假设性答案**两个增强步骤
- 检索采用了**稠密+稀疏混合向量**，兼顾语义相似度和关键词匹配
- 召回采用**多路并行**策略，用 RRF 融合后还用 Reranker 二次精排

---

### Q4：RAG 系统中常见的检索失败原因有哪些？你项目中是如何应对的？

**理想答案：**

常见检索失败原因及对应解决方案：

| 问题 | 我项目的应对方案 |
|------|-----------------|
| **语义鸿沟**：问题与文档用词不同 | BGE-M3 稠密向量捕捉语义相似度 |
| **关键词遗漏**：特定术语匹配不到 | BGE-M3 稀疏向量（词粒度）保留关键词匹配能力 |
| **问题模糊**：用户口语化、指代不明 | LLM 重写问题 + 实体识别，消除指代歧义 |
| **知识库覆盖不全** | MCP 网络搜索作为外部知识补充 |
| **单路召回偏差** | 三路并行召回（向量+HyDE+Web）互相补充 |
| **召回太多噪音文档** | RRF 融合 + Reranker 精排 + 断崖截断 |
| **文档切分不合理** | 按 Markdown 标题语义切分 + 长切短合策略 |

---

## 3. LangGraph 工作流类

### Q5：为什么选择 LangGraph 来编排流程，而不是自己写顺序调用代码？

**理想答案：**

LangGraph 是一个基于**有向状态图**的工作流编排框架，特别适合 RAG 这种有分支、有条件路由的多步骤流程。

**选择 LangGraph 的原因：**

1. **条件分支天然支持**：比如导入流程中，根据文件类型（PDF / MD）走不同的解析路径，LangGraph 的 `add_conditional_edges` 可以优雅实现。
2. **状态管理集中化**：所有节点共享一个 `TypedDict` 定义的状态字典（如 `ImportGraphState`、`QueryGraphState`），数据流转清晰可追溯。
3. **并行执行能力**：查询流程中三路召回（向量检索、HyDE 检索、MCP 搜索）是天然并行的，LangGraph 支持自动并行。
4. **流式执行**：`graph.stream(state)` 支持按节点输出中间结果，方便监控和调试。
5. **可维护性**：新增或调整流程节点时，只需修改图的边和节点，不需要改动业务逻辑代码。

**我的项目中有两个独立的 StateGraph：**
- `ImportGraphState`：管理导入流程的 15+ 个字段
- `QueryGraphState`：管理查询流程的 14+ 个字段

---

### Q6：导入流程中有一个条件路由 `route_after_entry`，说说它的设计思路？

**理想答案：**

```python
def route_after_entry(state: ImportGraphState):
    if state['is_md_read_enabled']:
        return "node_md_img"
    elif state['is_pdf_read_enabled']:
        return "node_pdf_to_md"
    else:
        return END
```

**设计思路：**

这个条件路由在入口节点后执行，根据文件类型选择不同的解析路径：

- **PDF 文件** → 先走 `node_pdf_to_md`（调用 MinerU 解析 PDF 为 Markdown），再统一汇入 `node_md_img`
- **Markdown 文件** → 直接走 `node_md_img`（处理 MD 中的图片）
- **其他格式** → 直接终止，避免无效处理

不管走哪条分支，最终都会汇集到 `node_md_img → node_document_split → ... → node_import_milvus` 这条统一流水线。这是一个典型的**分支-合并**模式，复用后续节点。

---

### Q7：查询流程中 `node_item_name_confirm` 节点是如何工作的？为什么要在检索前做这一步？

**理想答案：**

这是整个查询流程的**第一道关卡**，核心作用是：

1. **结合历史对话 + 当前问题 → 让 LLM 提取 item_names 并重写问题**
   - 利用历史上下文消除指代歧义（"它"、"那个" → 具体商品名）
   - 将口语化表达规范化

2. **向量库验证 item_names**
   - LLM 提取的商品名可能不精确（如"华为P60" vs 库里的"HUAWEI P60 Pro"）
   - 将提取的 item_name 转为向量，在 Milvus 的 `item_name_collection` 中做混合检索验证

3. **分级处理结果**
   - **优先级1**：名字完全相同 → 直接确认（不限分数）
   - **优先级2**：向量相似度 ≥ 0.85 → 确认
   - **优先级3**：向量相似度 ≥ 0.60 → 可选（最多2个，返回提示让用户选择）
   - **优先级4**：无匹配 → 直接返回"没有匹配的商品名"

4. **如果确认了 item_name → 继续后续三路检索；如果只有可选项或没有 → 直接返回 answer，提前终止**

**为什么要在检索前做这一步：**
- 作为**过滤器**，过滤掉无法准确识别的查询，避免浪费后续检索资源
- 确定的 item_name 可作为 Milvus 的搜索过滤条件（`expr=f"item_name in [... ]"`），大幅缩小检索范围，提升精确度

---

## 4. 向量数据库与检索类

### Q8：Milvus 中为什么使用稠密向量 + 稀疏向量的混合检索？各有什么优势？

**理想答案：**

**稠密向量（Dense Vector）**：
- BGE-M3 生成 1024 维浮点数向量
- 捕捉**语义相似度**：即使用词完全不同，只要语义相近就能匹配
- 例："怎么换电池" 能匹配到 "电量耗尽如何更换电源模块"

**稀疏向量（Sparse Vector）**：
- BGE-M3 同时生成的词粒度向量
- 捕捉**关键词精确匹配**：保留术语、型号等关键信息的字面匹配能力
- 例："HAK-180" 精确匹配到产品手册中的 "HAK-180 烫金机"

**混合检索优势**：
- 稠密负责"找意思相近的"，稀疏负责"找出现过的"
- 二者互补，防止语义漂移和关键词遗漏
- 通过 `WeightedRanker` 加权融合，我们设定稠密 0.65、稀疏 0.35 的权重（稠密为主，稀疏为辅）

```python
# 核心代码片段
def create_hybrid_search_requests(dense_vector, sparse_vector, ...):
    dense_req = AnnSearchRequest(data=[dense_vector], anns_field="dense_vector", ...)
    sparse_req = AnnSearchRequest(data=[sparse_vector], anns_field="sparse_vector", ...)
    return [dense_req, sparse_req]

def hybrid_search(client, collection_name, reqs, ranker_weights=(0.65, 0.35), ...):
    rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=True)
    res = client.hybrid_search(collection_name=collection_name, reqs=reqs, ranker=rerank, ...)
```

---

### Q9：Milvus 的索引参数是如何选择的？HNSW 的 M 和 efConstruction 参数怎么调？

**理想答案：**

在 `node_import_milvus.py` 中，稠密向量使用 **HNSW 索引**，稀疏向量使用 **SPARSE_INVERTED_INDEX**。

**稠密向量 (HNSW) 参数选择：**

```python
index_params.add_index(
    field_name="dense_vector",
    index_type="HNSW",       # 分层可导航小世界图——目前最主流的近似最近邻算法
    metric_type="COSINE",    # 余弦相似度，BGE-M3 已做 L2 归一化，等效于内积
    params={
        "M": 32,             # 每个节点最大连接数
        "efConstruction": 300 # 构建时搜索宽度
    }
)
```

**参数调优思路：**

| 参数 | 含义 | 权衡 |
|------|------|------|
| **M** | 每个节点最多连接的邻居数 | 越大 → 图更密 → 召回越高，但索引内存越大、构建越慢 |
| **efConstruction** | 构建时每次插入向量搜索的候选邻居数 | 越大 → 索引质量越高，但构建越慢 |

**经验法则：**
- 10K 向量 → M=16, ef=200
- 50K 向量 → M=32, ef=300（我的项目用这个）
- 100K 向量 → M=64, ef=400

**稀疏向量索引：**
使用 `SPARSE_INVERTED_INDEX`（倒排索引），算法设为 `DAAT_MAXSCORE`——一种高性能 TopK 剪枝检索算法，能提前估算文档得分并跳过低分文档。

---

### Q10：导入流程中如何保证幂等性（重复导入同一份文档不会产生重复数据）？

**理想答案：**

在 `node_import_milvus.py` 的 `step_3_delete_old_data` 方法中实现了幂等性：

```python
def step_3_delete_old_data(milvus_client, item_name):
    milvus_client.load_collection(collection_name=milvus_config.chunks_collection)
    milvus_client.delete(
        collection_name=milvus_config.chunks_collection,
        filter=f"item_name=='{item_name}'"
    )
```

**设计思路：**
- 以 `item_name`（文档对应产品的识别名）作为幂等键
- 先删后插策略：插入新数据前，先删除该 `item_name` 下的所有旧 chunks
- 这样即使同一份文档导入多次，向量库中也不会出现重复数据

---

## 5. 多路召回与融合排序类

### Q11：为什么要设计三路并行召回？HyDE 的原理是什么？

**理想答案：**

**三路召回的设计目的：**

| 召回路径 | 策略 | 优势 | 劣势 |
|---------|------|------|------|
| 向量检索 | 重写后的问题直接向量检索 | 语义匹配准确 | 依赖向量质量 |
| HyDE 检索 | LLM 先生成假设性答案，再用"问题+答案"检索 | 能召回更多相关文档 | 多一次 LLM 调用，耗时长 |
| MCP 网络搜索 | 调用外部搜索 API | 补充知识库没有的信息 | 外部结果质量不可控 |

三路召回互相补充，降低单路召回偏差的风险。

**HyDE (Hypothetical Document Embeddings) 原理：**

1. 先让 LLM 根据问题**先生成一个假设性的答案**（哪怕这个答案不完全准确）
2. 将"问题 + 假设性答案"拼接后向量化
3. 用这个更"丰富"的向量去检索知识库

**为什么有效？**
- 用户问题通常很短，向量化后信息量有限
- 假设性答案将问题"展开"成了接近文档风格的文本
- 检索时是在**文档空间**中匹配，而非问题空间中匹配

```python
def step_1_create_hyde_doc(rewritten_query):
    llm_client = get_llm_client()
    prompt = load_prompt("hyde_prompt", rewritten_query=rewritten_query)
    response = llm_client.invoke([HumanMessage(content=prompt)])
    return response.content  # 返回假设性答案
```

---

### Q12：说说 RRF（Reciprocal Rank Fusion）算法的原理和实现？

**理想答案：**

**RRF 倒排排名融合算法**：将多个检索源的排序结果融合成一个综合排名，不依赖原始分数的绝对大小，只依赖文档在各路中的排名位置。

**核心公式：**

```
RRF_score(d) = Σ( weight_i / (rank_i(d) + k) )
```

其中：
- `d` 是某个文档
- `weight_i` 是第 i 路召回源的权重
- `rank_i(d)` 是文档 d 在第 i 路中的排名（从 1 开始）
- `k` 是平滑常数（通常取 60），防止排名靠后的文档得分过低

**我的实现（`node_rrf.py`）：**

```python
def step_3_reciprocal_rank_fusion(source_with_weights, topk=5):
    score_dict = {}   # {chunk_id: RRF_score}
    chunk_dict = {}   # {chunk_id: chunk}

    for source, weight in source_with_weights:
        for rank, chunk in enumerate(source, start=1):
            chunk_id = chunk.get("entity").get("chunk_id")
            # RRF 公式
            score_dict[chunk_id] = score_dict.get(chunk_id, 0.0) + (1.0 / (rank + 60.0)) * weight
            chunk_dict[chunk_id] = chunk

    # 按 RRF 分数排序取 Top-K
    merge = sorted([(chunk_dict[cid], score) for cid, score in score_dict.items()],
                   key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in merge[:topk]]
```

**为啥用 RRF 而不是直接按分数排序？**
- 不同召回源的分数分布差异大（向量的余弦距离 vs 网络搜索的 BM25 分）
- RRF 只关注排名，避免了分数归一化的问题

---

### Q13：Reranker 精排和断崖截断算法是如何设计的？

**理想答案：**

**Reranker 的作用：**

RRF 融合后的结果仍然是基于"检索阶段"的粗糙排序。Reranker（我使用 BGE-Reranker-Large）是一个 **Cross-encoder 模型**，对每个 (query, document) 对进行逐对精细打分。

```python
# 构建 (问题, 文档) 对，逐个打分
input_pairs = [[rewritten_query, text] for text in text_list]
scores = rerank.compute_score(input_pairs, normalize=True)
```

Cross-encoder vs Bi-encoder（BGE-M3）的区别：
- Bi-encoder：问题和文档分别编码，然后计算相似度 → 快但粗糙
- Cross-encoder：问题和文档拼接后一起编码 → 慢但精准

**断崖截断算法：**

排名越靠后的文档，其与问题的相关性通常越低。如果相邻两个文档的分数差距突然变大（出现"断崖"），说明从这一位开始的文档基本不相关了，应该截断。

```python
for index in range(min_topk - 1, topk - 1):
    score1 = rerank_score_list[index].get("score")
    score2 = rerank_score_list[index + 1].get("score")
    gap = score1 - score2

    # 绝对阈值：分差 ≥ 0.5 直接截断
    # 相对阈值：分差/前分 ≥ 0.25 截断（防止 0.8, 0.7, 0.6... 一直触发不了绝对阈值）
    rela = gap / (abs(score1) + 1e-8)
    if gap >= abs_gap or rela >= rela_gap:
        topk = index + 1
        break
```

**两个阈值互补设计：**
- **绝对阈值 0.5**：处理高分区域的断崖（如 0.95 → 0.3）
- **相对阈值 0.25**：处理低分区域的断崖（如 0.3 → 0.2，0.5 绝对阈值永远触发不了，但 0.1/0.3=33% > 25%，触发截断）

---

## 6. Embedding 与模型类

### Q14：BGE-M3 模型的特点是什么？为什么选择它？

**理想答案：**

BGE-M3 是 BAAI（北京智源研究院）发布的多语言嵌入模型，核心特点是**一个模型同时输出稠密向量和稀疏向量**：

1. **Dense 输出**：1024 维浮点数稠密向量，用于语义相似度匹配
2. **Sparse 输出**：词粒度的稀疏向量（带权重的词索引字典），用于关键词精确匹配
3. **多语言支持**：支持中英等 100+ 种语言
4. **8192 Token 上下文窗口**：可以处理较长的文档片段

**选择 BGE-M3 的理由：**
- 一个模型承担两个角色，不需要分别部署稠密和稀疏模型
- 原生支持 L2 归一化（`normalize_embeddings=True`），适配 Milvus IP 内积检索
- 中文效果好，适合我们的产品手册、操作指南等中文文档场景
- 开源免费，可以在本地 GPU 上部署

---

### Q15：为什么在调用 LLM 时使用 JSON Mode？有什么注意事项？

**理想答案：**

在 `node_item_name_confirm` 中，LLM 需要返回 `{item_names: [...], rewritten_query: "..."}` 这样的结构化数据。为了保证解析可靠，我使用了三个保障措施：

1. **`response_format = {"type": "json_object"}`**：告诉模型必须返回合法 JSON
2. **Prompt 中明确格式要求 + 提供 Few-shot 示例**：让模型知道期望的 JSON schema
3. **返回后做容错解析**：处理模型偶尔会包裹 `json` 的情况

```python
# 设置 JSON Mode
llm_client = get_llm_client(json_mode=True)

# 容错解析
content = response.content
if content.startswith("```json"):
    content = content.replace("```json", "").replace("```", "")
dict_content = json.loads(content)

# 字段兜底
if "item_names" not in dict_content:
    dict_content["item_names"] = []
if "rewritten_query" not in dict_content:
    dict_content["rewritten_query"] = original_query
```

**注意事项：**
- JSON Mode 不影响模型的理解能力，只约束输出格式
- 不是所有模型都支持 JSON Mode，需要检查 API 兼容性
- 始终要做好解析失败后的兜底处理

---

### Q16：Embedding 处理时为什么要做批量处理（batch_size=5）？

**理想答案：**

在 `node_bge_embedding.py` 中：

```python
for i in range(0, len(chunks), batch_size):
    batch_chunks = chunks[i:i+batch_size]
    current_text = [f"商品名：{chunk.get('item_name')},介绍：{chunk.get('content')}" for chunk in batch_chunks]
    result = generate_embeddings(current_text)
```

**批量处理的原因：**

1. **GPU 利用率**：单条文本向量化时 GPU 利用率很低，批量处理可以充分利用 GPU 并行计算能力
2. **内存控制**：BGE-M3 的上下文窗口是 8192 tokens，如果 chunk 内容较长，一次加载太多会 OOM（显存溢出）。batch_size=5 是根据实际 chunk 长度和 GPU 显存调出来的经验值
3. **加速向量化**：文本生成向量时，将 item_name 和 content 拼接为 `"商品名：{item_name},介绍：{content}"`，让向量同时包含商品名和内容信息，增强检索时的语义匹配

---

## 7. 工程实践与架构类

### Q17：项目中大量使用了单例模式（Milvus 客户端、BGE-M3 模型、LLM 客户端等），为什么？单例有什么风险？

**理想答案：**

**为什么使用单例：**

1. **避免重复初始化开销**：BGE-M3 模型加载需要几秒到几十秒，占用数 GB 显存；Milvus 连接建立也有网络开销
2. **资源复用**：LLM 客户端背后是 HTTP 连接池，单例避免重复创建连接
3. **全局统一管理**：配置集中，状态一致

```python
# BGE-M3 单例模式
_bge_m3_ef = None

def get_bge_m3_ef():
    global _bge_m3_ef
    if _bge_m3_ef is not None:
        return _bge_m3_ef
    _bge_m3_ef = BGEM3EmbeddingFunction(...)
    return _bge_m3_ef
```

**单例的风险和应对：**

- **线程安全问题**：多线程并发初始化可能导致创建多个实例。Python 的 GIL 提供了部分保护，但在高并发场景下最好加锁（`threading.Lock`）
- **连接失效问题**：Milvus 连接可能因网络抖动断开，此处缺少断线重连机制——这是一个可改进点
- **测试困难**：单例使得单元测试难以 mock，可考虑依赖注入作为替代

---

### Q18：项目中如何处理流式和非流式两种响应模式？

**理想答案：**

系统同时支持两种响应模式，通过 `is_stream` 参数切换：

**流式模式（SSE）：**

```
前端 POST /query (is_stream=true) → 后台异步执行 LangGraph → 
中间结果通过 SSE 队列推送 → 前端 EventSource 接收实时更新
```

```python
# 非流式：直接调用 invoke
response = llm_client.invoke(prompt)
answer = response.content

# 流式：逐块 stream + SSE 推送
for chunk in llm_client.stream(prompt):
    delta = chunk.content
    answer += delta
    push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})
```

**流式的优势：**
- 用户能实时看到 LLM 逐字生成答案，体验好（类似 ChatGPT）
- 避免了长时间等待产生的焦虑

**非流式的优势：**
- 简单直接，适合一次性返回完整答案的场景
- 不需要维护 SSE 长连接

**实现细节：**
- 流式请求使用 FastAPI 的 `BackgroundTasks`，接口立即返回 `session_id`
- 前端通过 `GET /stream/{session_id}` 建立 SSE 长连接接收事件
- 系统还推送了 `progress`（节点进度）、`FINAL`（最终结果含图片 URL）、`ERROR`（异常）等多种 SSE 事件类型

---

### Q19：MongoDB 在项目中扮演什么角色？索引是怎么设计的？

**理想答案：**

MongoDB 用于存储**对话历史记录**，核心使用场景：

1. **历史上下文注入**：每次新提问时，查询最近 10 条对话记录，作为 LLM 的上下文
2. **实体识别辅助**：历史记录中的 `item_names` 帮助 LLM 在多轮对话中保持实体一致性
3. **对话记录持久化**：每次对话结束后，存储 user 和 assistant 的消息

**索引设计：**

```python
# 复合索引：session_id 升序 + ts 降序
self.chat_message.create_index([("session_id", 1), ("ts", -1)])
```

**设计原因：**
- 查询场景是"按会话查询，按时间排序"
- `session_id` 在前：快速定位到指定会话
- `ts` 降序：便于获取最新记录（覆盖"最近 N 条"的核心查询场景）
- 复合索引一次查询即可完成 `find + sort`，无需 filesort

---

### Q20：项目中 MinIO 的作用是什么？存储桶策略是如何配置的？

**理想答案：**

MinIO 是兼容 S3 协议的**对象存储服务**，在项目中用于：

1. **存储文档中提取的图片**：PDF/MD 文件中的图片被提取后上传到 MinIO
2. **提供公网可访问的图片 URL**：生成的答案中需要展示相关图片时，直接引用 MinIO URL

**存储桶策略：**

```python
bucket_policy = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"AWS": ["*"]},      # 所有匿名用户
        "Action": ["s3:GetObject"],        # 仅允许读取
        "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
    }]
}
minio_client.set_bucket_policy(bucket_name, json.dumps(bucket_policy))
```

**设计考虑：**
- 前端展示图片时直接使用 MinIO 的 HTTP URL，无需经过后端转发
- `s3:GetObject` 只读权限，不允许上传/删除，保证安全性
- 内网部署 `secure=False` 使用 HTTP，简化配置

---

### Q21：MCP（Model Context Protocol）在你的项目中是如何使用的？

**理想答案：**

MCP 是 Anthropic 提出的模型上下文协议，用于标准化 LLM 与外部工具的交互。我项目中通过 MCP 协议接入了**阿里云百炼的 WebSearch 工具**，作为 RAG 的外部知识补充。

**实现流程：**

```python
async def mcp_call_streamble(query):
    # 1. 创建 MCP SSE 客户端（连接百炼 MCP 服务）
    search_mcp = MCPServerStreamableHttp(
        name='search_mcp',
        params={
            "url": DASHSCOPE_BASE_URL_STREAMBLE,
            "headers": {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
            "timeout": 10
        },
        max_retry_attempts=3  # 最多重试3次
    )

    # 2. 连接 → 发现工具 → 调用 → 关闭
    await search_mcp.connect()
    result = await search_mcp.call_tool(
        tool_name="bailian_web_search",
        arguments={"query": query, "count": 5}
    )
    return result
```

**MCP 带来的好处：**
- 标准化的工具调用接口，切换不同搜索服务只需改 URL 和凭证
- 支持自动发现可用工具（`list_tools()`）
- 超时控制和重试机制，保证服务稳定性

---

## 8. 系统设计与扩展类

### Q22：如果文档量从 100 篇增加到 10 万篇，系统哪些地方会遇到瓶颈？如何优化？

**理想答案：**

**可能的瓶颈和优化方案：**

| 维度 | 瓶颈 | 优化方案 |
|------|------|---------|
| **向量检索** | Milvus 单机 HNSW 索引检索延迟随数据量增长 | 1. Milvus 分布式部署（Proxy + QueryNode + DataNode）2. 分区（partition）按 item_name 分片 3. 粗排+精排两阶段检索 |
| **Embedding 生成** | BGE-M3 单 GPU 吞吐量有限 | 1. 多 GPU 模型并行 2. 异步批量处理队列 3. 预向量化缓存 |
| **MongoDB 历史记录** | 单集合数据膨胀，查询变慢 | 1. 按时间分片（Sharding）2. 冷数据归档 3. TTL 索引自动清理过期记录 |
| **文档导入** | 串行导入 10 万篇耗时过长 | 1. 并行导入多个文档 2. 异步任务队列（Celery/Redis） |
| **LLM 调用** | API 限流 + 高并发下延迟增加 | 1. 本地部署开源模型（vLLM/TGI）2. 请求排队+限流 3. 常见问题缓存答案 |

---

### Q23：如果要给系统加入多轮对话能力，你会怎么设计？

**理想答案：**

实际上我的项目**已经实现了基础的多轮对话能力**：

**现有实现：**
1. MongoDB 存储历史对话记录（最近 10 条）
2. LLM 问题重写时参考历史上下文（消除"它"、"那个"等指代）
3. 生成答案时历史对话作为 Prompt 的一部分注入

**如果要进一步增强，我会：**

1. **对话状态管理**：维护会话级别的 `ConversationState`，记录当前讨论的实体、话题等
2. **滑动窗口 + 摘要**：对话超过一定轮次后，对早期对话做摘要压缩，避免 Context Window 溢出
3. **追问检测**：识别用户的追问意图（"还有吗？"、"具体怎么操作？"），自动在前一轮检索结果基础上做二次检索
4. **澄清机制**：当检测到用户意图模糊时，主动发问澄清（类似系统中已有的"可选 item_name"提示）

---

### Q24：如何评估你这个 RAG 系统的效果？你会设计哪些指标？

**理想答案：**

**离线评估指标：**

| 指标 | 定义 | 计算方式 |
|------|------|---------|
| **Recall@K** | Top-K 检索结果中包含正确答案的比例 | 标准答案是否在 Top-K 个 chunks 中 |
| **MRR** (Mean Reciprocal Rank) | 第一个正确答案排名的倒数 | 1/第一个正确答案的排名，取平均 |
| **NDCG@K** | 归一化折损累计增益 | 考虑排序位置权重的命中率 |
| **Faithfulness** | 生成答案是否忠实于检索文档 | 人工/LLM 评判答案能否从检索文档中推出 |
| **Answer Correctness** | 生成答案的正确性 | 与标准答案的语义相似度或人工评分 |

**在线评估：**
- 用户点赞/点踩率
- 用户追问率（追问越多说明首次回答不够好）
- 平均对话时长/轮次

**我的做法：**
- 构建测试集（问题-标准答案对），用 `node_rrf` 和 `node_rerank` 的本地测试代码中的 `if __name__ == "__main__"` 模块做单元测试
- 调整 retriever 和 reranker 参数时对比 MRR 变化

---

## 9. 场景题 / 故障排查类

### Q25：用户反馈"系统经常检索不到相关内容"，你会怎么排查？

**理想答案：**

按以下链路逐层排查：

1. **问题理解层**（`node_item_name_confirm`）
   - LLM 是否正确提取了 item_name？
   - 问题重写是否合理？
   - 查看日志：`logger.info(f"LLM 提取结果 ... item_names: ... rewritten_query: ...")`
   - 如果 item_name 提取错误或为空 → 优化 item_name_recognition 的 prompt

2. **向量检索层**（`node_search_embedding`）
   - 向量化是否正常？检查 `generate_embeddings` 的输入输出
   - Milvus 是否正常返回结果？检查 `response` 是否为空
   - expr 过滤条件是否过严？检查 `item_name in [...]` 是否正确构造

3. **文档切分层**（`node_document_split`）
   - Chunk 是否被切得太碎或太长？检查 `chunks` 的内容
   - 关键信息是否被截断？调整 `DEFAULT_MAX_CONTENT_LENGTH` 和 `MIN_CONTENT_LENGTH`

4. **融合排序层**（`node_rrf` + `node_rerank`）
   - 三路召回各自返回多少结果？
   - RRF 权重是否合理？
   - 断崖截断是否过于激进？调整 `RERANK_GAP_RATIO` 和 `RERANK_GAP_ABS`

5. **生成层**（`node_answer_output`）
   - Prompt 是否正确组装了检索到的 chunks？
   - LLM 是否忽略了检索结果（幻觉）？

**排查工具：**
- 项目中每个节点都有 `logger.info` 日志，可以直接追踪
- 每个节点的 `if __name__ == "__main__"` 提供了本地单元测试入口

---

### Q26：如果 Milvus 服务突然挂了，系统会怎样？如何改进？

**理想答案：**

**现有问题：**
当前代码在 Milvus 连接失败时，`get_milvus_client()` 返回 `None`，但后续 `hybrid_search` 等方法没有对 `None` 做充分防御，可能导致 `AttributeError`。

**改进方案：**

1. **连接重试机制**：
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_milvus_client():
    ...
```

2. **服务降级**：Milvus 挂了时，降级只使用 MCP 网络搜索结果
3. **健康检查**：`/health` 接口增加 Milvus 连接状态检查
4. **告警通知**：连接失败时通过日志告警（`logger.error` 已有，可接入监控系统）

---

## 10. 行为与软技能类

### Q27：在这个项目中你遇到的最大技术挑战是什么？怎么解决的？

**建议回答思路（结合项目实际）：**

1. **挑战**：多路召回结果的融合排序——三路召回的分数分布差异很大，直接按原始分合并效果差
2. **解决过程**：
   - 先尝试直接按分数排序 → 效果差，各路分数不可比
   - 调研后选择 RRF（倒数排名融合），只关注排名不关注绝对分数
   - RRF 后再加 Reranker（Cross-encoder）做精排
   - 发现 Reranker 后仍有低分噪音 → 引入断崖截断算法
3. **结果**：通过 RRF + Reranker + 断崖截断三级排序，最终检索准确率显著提升

### Q28：如果让你重新设计这个系统，你会做哪些改进？

**理想答案：**

1. **引入依赖注入**：当前大量使用全局变量和单例模式，测试困难。改用 FastAPI 的依赖注入或自定义 DI 容器
2. **增加可观测性**：接入 OpenTelemetry，追踪每个节点的耗时（当前只有日志，缺少分布式追踪）
3. **检索缓存**：对高频问题做答案缓存（Redis），减少 LLM 调用和向量检索开销
4. **A/B 测试框架**：对不同检索策略做 A/B 测试，用数据驱动优化（当前只能手动调参对比）
5. **文档处理增强**：支持更多格式（Word、PPT、网页等），当前只支持 PDF 和 MD
6. **向量数据库切换能力**：抽象 VectorStore 接口，支持切换 Qdrant、Weaviate 等
7. **断线重连与连接池**：Milvus、MongoDB、MinIO 的连接缺少重连机制

---

## 快速复习清单

面试前可以快速过一遍的项目关键数据：

| 项目 | 内容 |
|------|------|
| **技术栈** | FastAPI + LangGraph + Milvus + BGE-M3 + BGE-Reranker + MongoDB + MinIO + MCP |
| **两个 StateGraph** | ImportGraphState (15 字段) + QueryGraphState (14 字段) |
| **导入流程节点（7个）** | entry → pdf_to_md → md_img → document_split → item_name_recognition → bge_embedding → import_milvus |
| **查询流程节点（7个）** | item_name_confirm → (向量检索 ‖ HyDE ‖ MCP) → rrf → rerank → answer_output |
| **三路召回** | Dense+Sparse 混合向量检索 / HyDE 假设性答案 / MCP 网络搜索 |
| **三级排序** | 混合检索(粗排) → RRF(融合) → Reranker(精排) → 断崖截断 |
| **Embedding 维度** | 1024 维稠密 + 不定长稀疏向量 |
| **向量检索模式** | 稠密 COSINE + 稀疏 IP（内积），混合权重 0.65:0.35 |
| **Milvus 索引** | 稠密: HNSW (M=32, ef=300) / 稀疏: SPARSE_INVERTED_INDEX (DAAT_MAXSCORE) |
| **LLM 服务** | 阿里云百炼 DashScope（千问系列），兼容 OpenAI API |
| **流式响应** | SSE (Server-Sent Events)，4 种事件类型：DELTA / FINAL / ERROR / PROGRESS |
| **幂等策略** | 按 item_name 先删后插 |

---

> **面试提醒**：面试官最喜欢问的不是"你用了什么"，而是"你为什么这样用"和"你遇到了什么问题，怎么解决的"。准备好每个技术选型的"为什么"和你自己的思考过程，比背答案重要得多。祝面试顺利！🎯
