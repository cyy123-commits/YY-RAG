import sys


from app.utils.task_utils import add_running_task,add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import create_hybrid_search_requests,hybrid_search,get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())


def node_search_embedding(state):
    """
    节点功能:进行向量内容检索
    主要作痛：问题-》查询chunks切片
    达到目标返回：{”embedding_chunks“:[chunks]}
    主要用到的参数：{
                rewritten_query:str
                item_names:[]
                }
    :param state:
    :return:
    """
    logger.info(f"进行混合向量内容检索")
    add_running_task(state['session_id'],sys._getframe().f_code.co_name,state.get("is_stream"))




    #1.获取状态里的参数
    rewritten_query=state.get("rewritten_query")
    item_names=state.get("item_names")

    #2。将重写问题进行向量化生成稀疏和稠密矩阵
    embedding=generate_embeddings([rewritten_query])
    #3。查询向量数据库 chunks那个表  重写问题与每个chunk的向量进行混合检索
    #3.1创建混合查询对象AnnSearchRequset

    # 构造 Milvus 过滤表达式：item_name in ["A", "B"]（双引号为 Milvus 规范）
    #f"item_name in {json.dumps(item_names, ensure_ascii=False)}"或者可以直接用json,dumps
    items_str = '["' + '", "'.join(item_names) + '"]'
    reqs=create_hybrid_search_requests(
        dense_vector=embedding["dense"][0],
        sparse_vector=embedding["sparse"][0],
        expr = f"item_name in {items_str}"
    )
    #3.2进行混合检索
    response=hybrid_search(
        client=get_milvus_client(),
        collection_name=milvus_config.chunks_collection,
        reqs=reqs,
        ranker_weights=(0.65,0.35),
        norm_score=True,
        limit=5,
        output_fields=["chunk_id","content","title","file_title","parent_title","item_name"]

    )
    #4.处理查询结果，赋值embedding_chunks属性
    embedding_chunks=response[0] if response else []
    # [
    # {id:, distance:, entity: {:}  }，
    # {id:, distance:, entity: {:}  }，
    # {id:, distance:, entity: {:}  }，
    # {id:, distance:, entity: {:}  }，
    # {id:, distance:, entity: {:}  }
    # ]

    add_done_task(state["session_id"],sys._getframe().f_code.co_name,state.get("is_stream"))

    logger.info(f"混合向量内容检索结束")

    return {"embedding_chunks":embedding_chunks}





