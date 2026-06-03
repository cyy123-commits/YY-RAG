import sys
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


def step_3_reciprocal_rank_fusion(source_with_weights,topk:int=5):
    """
    节点核心方法，用rrf算法对同源不同路结果进行重排，得到分数最高的五个

    :param source_with_weights:  [(embedding_chunks,weight),(hype_embedding_chunks,weight)]
    :param topk: 取rrf分数最高的k个chunk
    :return:[chunks] 这里说的chunk是  {id：，distance:,entity:{chunk_id:,content:,.....}}
    """

    score_dict={}#存储{chunk_id:score,chunk_id:score,....}
    chunk_dict={}#存储{chunk_id:chunk.chunk_id:chunk,....}
    #后面根据这两个字典按照chunk_id来形成[(chunk，score).(chunk,score),....],=
    #对他排序获取前k个chunk，存入最终列表

    #遍历两路数据
    for source,weight in source_with_weights:

        for rank,chunk in enumerate(source,start=1):
            chunk_id=chunk.get("entity").get("chunk_id")

            #rrf算法计算每个chunk的分数，
            score_dict[chunk_id]=score_dict.get(chunk_id,0.0)+(1.0/(rank+60.0))*weight
            #将chunk放入对应的chunk_id下
            chunk_dict[chunk_id]=chunk
    merge=[]
    for chunk_id,score in score_dict.items():
        chunk=chunk_dict.get(chunk_id)
        merge.append((chunk,score))
    merge.sort(key = lambda x:x[1],reverse=True)
    merge_ranked=merge[:topk]

    #最后只要chunk就行，不用保留分数
    rank_chunks=[chunk for chunk,score in merge_ranked]

    logger.info(f"完成了rrf排序，结果为{rank_chunks}")
    return rank_chunks




def node_rrf(state):
    """
    对多路召回的结果（向量，hyde，web）进行融合排血，使用后rrf算法

    :param state:
    :return:
    """
    print("------RRF---------")
    node_name=sys._getframe().f_code.co_name
    add_running_task(state["session_id"],node_name, state["is_stream"])

    #获取参数   这里参数是前两个节点保存到state中的embedding_chunks和hyde_embedding_chunks
    embedding_chunks=state.get("embedding_chunks")
    hyde_embedding_chunks=state.get("hyde_embedding_chunks")

    #2.把这两路数据整合到一起
    source_with_weights=[
        (embedding_chunks,0.55),
        (hyde_embedding_chunks,0.45)
    ]

    #3.rrf算法进行数据重排序
    rrf_result=step_3_reciprocal_rank_fusion(source_with_weights)
    #结果结构：[{id:,distance,entity:{:::}},{},{}]  一堆chunk



    #将融合结果存入state中
    state["rrf_chunks"]=rrf_result
    add_done_task(state["session_id"],node_name, state["is_stream"])

    return state




if __name__ == "__main__":
    print("\n" + "="*50)
    print(">>> 启动 node_rrf 本地测试")
    print("="*50)

    # 1. 构造假数据 (模拟真实数据库字段)
    # 模拟 Embedding 检索结果
    mock_embedding_chunks = [
        {
            "id": "doc_1",
            "pk": "pk_1",
            "file_title": "操作手册_v1.pdf",
            "item_name": "HAK 180 烫金机",
            "content": "内容1：打开电源开关...",
            "score": 0.9
        },
        {
            "id": "doc_2",
            "pk": "pk_2",
            "file_title": "维修指南.pdf",
            "item_name": "HAK 180 烫金机",
            "content": "内容2：遇到故障请联系...",
            "score": 0.8
        },
        {
            "id": "doc_3",
            "pk": "pk_3",
            "file_title": "参数表.xlsx",
            "item_name": "HAK 180 烫金机",
            "content": "内容3：电压220V...",
            "score": 0.7
        }
    ]

    # 模拟 HyDE 检索结果 (包含 3 个文档，顺序不同，且有新文档 doc_4)
    mock_hyde_chunks = [
        {
            "id": "doc_3",
            "pk": "pk_3",
            "file_title": "参数表.xlsx",
            "item_name": "HAK 180 烫金机",
            "content": "内容3：电压220V...",
            "score": 0.85
        },
        {
            "id": "doc_1",
            "pk": "pk_1",
            "file_title": "操作手册_v1.pdf",
            "item_name": "HAK 180 烫金机",
            "content": "内容1：打开电源开关...",
            "score": 0.82
        },
        {
            "id": "doc_4",
            "pk": "pk_4",
            "file_title": "安全须知.docx",
            "item_name": "HAK 180 烫金机",
            "content": "内容4：操作时请佩戴手套...",
            "score": 0.75
        }
    ]

    # 模拟输入状态
    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "embedding_chunks": mock_embedding_chunks,
        "hyde_embedding_chunks": mock_hyde_chunks
    }

    try:
        # 运行节点
        result = node_rrf(mock_state)

        # 验证结果
        rrf_chunks = result.get("rrf_chunks", [])
        print("\n" + "="*50)
        print(">>> 测试结果摘要:")
        print(f"输入数量: Embedding={len(mock_embedding_chunks)}, HyDE={len(mock_hyde_chunks)}")
        print(f"输出数量: {len(rrf_chunks)}")
        print("-" * 30)

        # 打印详细排名
        print("最终排名:")
        for i, doc in enumerate(rrf_chunks, 1):
            # 注意：返回结果中可能没有 chunk_id 字段，而是 id
            doc_id = doc.get('chunk_id') or doc.get('id')
            print(f"Rank {i}: ID={doc_id}, Title={doc.get('file_title')}, Content={doc.get('content')[:20]}...")

        # 验证预期逻辑：
        ids = [d.get("id") or d.get("chunk_id") for d in rrf_chunks]

        if "doc_1" in ids and "doc_3" in ids:
            print("\n[PASS] 交叉文档 (doc_1, doc_3) 成功融合保留")
        else:
            print("\n[FAIL] 交叉文档丢失")

        if len(ids) == 4:
            print("[PASS] 并集数量正确 (3+3-2重叠=4)")
        else:
            print(f"[FAIL] 并集数量错误: 期望4, 实际{len(ids)}")

        print("="*50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
