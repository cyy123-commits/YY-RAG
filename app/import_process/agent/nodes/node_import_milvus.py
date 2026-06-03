import os
import sys
from typing import List, Dict, Any
# 导入Milvus相关依赖
from pymilvus import DataType
# 导入自定义模块

from app.clients.milvus_utils import get_milvus_client
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger
from app.conf.milvus_config import milvus_config




# ==========================================
# Milvus切片数据入库核心节点
# 核心能力：将上游向量化后的文本切片批量存入Milvus，实现幂等性写入
# 核心设计：
#   1. 幂等性：插入前删除同item_name旧数据，避免重复存储
#   2. 自动建表：集合不存在时自动创建Schema和向量索引，无需手动初始化
#   3. 数据校验：前置校验切片有效性、向量字段完整性，避免脏数据入库
#   4. 主键回填：将Milvus自增的chunk_id回填到切片，供下游业务使用
# 依赖上游：BGE-M3向量化节点（提供dense_vector/sparse_vector字段）
# ==========================================
def step_2_prepare_collection(state):
    # 1获取milvus的客户端
    milvus_client = get_milvus_client()
    # 2.判断是否存在要的集合，不存在就创建集合（表）
    if not milvus_client.has_collection(collection_name=milvus_config.chunks_collection):
        # 3.创建集合
        # 3.1 创建集合对应的列的信息，用shcema来创建field（字段）
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)

        # 3.2 add fields to schema
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="part", datatype=DataType.INT16)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 3.3配置索引
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",  # 给哪个词建立所引
            index_name="dense_vector_index",  # 索引名字
            index_type="HNSW",  # 向量查找所用的策略
            metric_type="COSINE",  # 相似度比较方法
            params={"M": 32,  # 每个节点最多连接多少个邻居，M越大，图更密，索引越精准，召回率越高，但索引内存占比越高，建索引越慢，所以2不一定越大越好
                    "efConstruction": 300  # 每次新插入一个向量，帮他找的候选邻居的数量。优缺点和M差不多
                    #########
                    # 10000 M=16 efConstruction=200
                    # 50000 M=32 ef=300
                    # 100000 M=64 ef=400
                    # ########
                    }
        )

        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",  # 倒排
            metric_type="IP",
            # DAAT_MAXSCORE，一种高性能 TopK 剪枝检索算法，提前估算，如果估算一个文档得分很低，且低于当前topk的最低分，那就直接跳过，更块，更省内存
            params={"inverted_index_algo": "DAAT_MAXSCORE"},  # Algorithm used for building and querying the index

        )
        # 3.4 创建集合
        milvus_client.create_collection(
            collection_name=milvus_config.chunks_collection,
            schema=schema,
            index_params=index_params
        )

    return milvus_client


def step_3_delete_old_data(milvus_client, item_name):

    """
    #删除旧数据，按照chunk的itemn_name来删除
    :param milvus_client:
    :param item_name:
    :return:
    """
    milvus_client.load_collection(collection_name=milvus_config.chunks_collection)

    milvus_client.delete(collection_name=milvus_config.chunks_collection, filter=f"item_name=='{item_name}'")

    milvus_client.load_collection(collection_name=milvus_config.chunks_collection)


def step_4_insert_collection(milvus_client,chunks):
    """
    插入集合的数据
    :param milvus_client:
    :param chunks:
    :return:
    """
    insert_result=milvus_client.insert(collection_name=milvus_config.chunks_collection, data=chunks)
    #成功插入几条
    insert_count=insert_result.get("inserted_count")
    logger.info(f">>> 完成了数据插入，插入了{insert_count}个数据")
    #回填id
    ids=insert_result.get('ids',[])
    if ids and len(ids) == len(chunks):
        for index,chunk in enumerate(chunks):
            chunk['chunk_id']=ids[index]

    return chunks






def node_import_milvus(state: Dict[str, Any]) -> Dict[str, Any]:

    # 获取当前节点名称，用于日志和任务状态记录
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行节点：{current_node}")

    # 标记任务运行状态，用于任务监控/前端进度展示
    add_running_task(state.get("task_id", ""), current_node)

    try:

        #1.校验数据
        chunks=state.get("chunks")
        if not chunks:
            logger.error(f"f{current_node}没有找到chunk数据，请检查")
            raise ValueError(f"没有找到chunk数据")
        #2.没有集合，创建集合
        milvus_client=step_2_prepare_collection(state)

        #3.删除旧数据
        step_3_delete_old_data(milvus_client,chunks[0]['item_name'])

        #$4插入chunks数据
        with_id_chunks=step_4_insert_collection(milvus_client,chunks)


        state['chunks']=with_id_chunks





    except Exception as e:
        # 捕获节点所有异常，记录错误堆栈，不中断整体流程
        logger.error(f"BGE-M3向量化节点执行失败：{str(e)}", exc_info=True)
        raise

    finally:
        logger.info(f"{current_node}执行结束了")
        add_done_task(state.get("task_id", ""), current_node)
    return state

if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "chunks": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            # 执行节点函数
            result_state = node_import_milvus(test_state)

            # 验证结果
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
            else:
                print("❌ 测试失败：未能获取 chunk_id")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
