from typing_extensions import TypedDict
from typing import List
import copy
from app.core.logger import logger


class QueryGraphState(TypedDict):
    """
    QueryGraphState 定义了整个查询流程中流转的数据结构。
    """
    session_id: str  # 会话唯一标识
    original_query: str  # 用户原始问题

    # 检索过程中的中间数据
    embedding_chunks: list  # 普通向量检索回来的切片
    hyde_embedding_chunks: list  # HyDE 检索回来的切片

    web_search_docs: list  # 网络搜索回来的文档

    # 排序过程中的数据
    rrf_chunks: list  # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终 Top-K 文档

    # 生成过程中的数据
    prompt: str  # 组装好的 Prompt
    answer: str  # 最终生成的答案

    # 辅助信息
    item_names: List[str]  # 提取出的商品名称
    rewritten_query: str  # 改写后的问题
    history: list  # 历史对话记录
    is_stream: bool  # 是否流式输出标记

    image_urls:list

# 建议定一个初始化对象，方便后续使用
# 定义图状态的默认初始值
query_graph_default: QueryGraphState = {
    "session_id":"",
    "original_query":"",
    "embedding_chunks":[],
    "hyde_embedding_chunks":[],
    "web_search_docs":[],
    "rrf_chunks":[],
    "reranked_docs":[],
    "prompt":"",
    "answer":"",
    "item_names":[],
    "rewritten_query":"",
    "history":[],
    "is_stream":False,
    "image_urls":[]
}

def create_default_state(**overrides) -> QueryGraphState:
    """
    创建默认状态，支持覆盖

    Args:
        **overrides: 要覆盖的字段（关键字参数解包）

    Returns:
        新的状态实例

    Examples:
        state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """

    # 默认状态
    state = copy.deepcopy(query_graph_default)
    # 用 overrides 覆盖默认值
    state.update(overrides)
    # 返回创建好的状态字典实例
    return state

def get_default_state() -> QueryGraphState:
    """
    返回一个新的状态实例，避免全局变量污染
    """
    return copy.deepcopy(query_graph_default)
