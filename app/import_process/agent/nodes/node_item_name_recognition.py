# 导入基础库：系统、路径、类型注解（类型注解提升代码可读性和可维护性）
import os
import sys
from typing import List, Dict, Any, Tuple
from app.conf.milvus_config import milvus_config
# 导入Milvus客户端（向量数据库核心操作）、数据类型枚举（定义集合Schema）
from pymilvus import MilvusClient, DataType
# 导入LangChain消息类（标准化大模型对话消息格式）
from langchain_core.messages import SystemMessage, HumanMessage

# 导入自定义模块：
# 1. 流程状态载体：ImportGraphState为LangGraph流程的统一状态管理对象
from app.import_process.agent.state import ImportGraphState
# 2. Milvus工具：获取单例Milvus客户端，实现连接复用
from app.clients.milvus_utils import get_milvus_client
# 3. 大模型工具：获取大模型客户端，统一模型调用入口
from app.lm.lm_utils import get_llm_client
# 4. 向量工具：BGE-M3模型实例、向量生成方法（稠密+稀疏向量）
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
# 5. 稀疏向量工具：归一化处理，保证向量长度为1，提升检索准确性
from app.utils.normalize_sparse_vector import normalize_sparse_vector
# 6. 任务工具：更新任务运行状态，用于任务监控和管理
from app.utils.task_utils import add_running_task
# 7. 日志工具：项目统一日志入口，分级输出（info/warning/error）
from app.core.logger import logger
# 8. 提示词工具：加载本地prompt模板，实现提示词与代码解耦
from app.core.load_prompt import load_prompt


# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500


""" 
主要目标：1.挑去topk个chunk，利用大语言模型识别出对应的iten_name，用于区分不同文档
        2.使用嵌入式模型，将item_name生成的向量存储到向量数据库中
        3。修改state中的chunks，chunks中加入item_name-》{title,content,file_title,parent_title,part,item_name}
        
具体步骤：1.校验：       校验state中的file_title和chunks，file_title用于保底
        2.构建上下文：  挑去topk个chunks构建context
        3.调用llm：    利于大语言模型识别出chunks的item_name，识别不出就直接用file_title
        4.回填数据：    将item_name放到每个chunks中，更新state
        5.调用嵌入式模型：利用嵌入式模型将item_name生成稠密向量和稀疏向量
        6.向量数据库：   存储向量到向量数据库中kb_item_name（id,file_title_item_name,sparse_vector,dense_vector）
"""

# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500


def step_1_get_chunks(state):
    """
    获取状态中的chunks和file_title
    :param state: 全局状态
    :return: 返回chunks和file_title
    """
    chunks=state['chunks']
    file_title=state['file_title']

    if not chunks:
        raise ValueError(f"chunks中没有东西，无法继续进行程序")

    if not file_title:
        file_title=os.path.splitext(os.path.basename(state.get('md_path')))[0]
        state['file_title']=file_title
        logger.info(f"file_title缺失，从md_path中截取：{file_title}")
    return chunks,file_title



def step_2_bulid_context(chunks):
    """

    截取topk个chunks，将他们的conten拼接到一起  且有三个条件：1.截取前k个chunks，2.单个切片有长度限制  3.拼接起来的所有切片也有长度限制
    :param chunks:
    :return:
    返回格式设置成一下形式：一个列表中有多条数据  切片：{1}，标题：{title},内容：{content}
                                         切片：{2}，标题：{title},内容：{content}
                                         切片：{3}，标题：{title},内容：{content}
    """

    #前置工作
    total_chars=0 #存储起来的总的长度
    parts=[]#存储处理后的切片：[切片：{1}，标题：{title},内容：{content},切片：{2}，标题：{title},内容：{content},..]
    #循环处理content，同时要判断好长度
    for index,chunk in enumerate(chunks[:DEFAULT_ITEM_NAME_CHUNK_K],start=1):
        chunk_title=chunk['title']
        chunk_content=chunk['content'][:SINGLE_CHUNK_CONTENT_MAX_LEN]
        data=f"切片：{index}，标题：{chunk_title},内容：{chunk_content}"
        parts.append(data)

        total_chars+=len(data)

        if total_chars>=CONTEXT_TOTAL_MAX_CHARS:
            logger.info(f"达到了最大切片长度")
            break
    context='\n\n'.join(parts)
    final_context=context[:CONTEXT_TOTAL_MAX_CHARS]

    return final_context


def step_3_call_llm(context, file_title):
    """
    调用大语言模型，过去item_name,如果获取不到使用file_title兜底

    :param context:
    :param file_title:
    :return:
    """
    #1加载提示词
    human_prompt=load_prompt("item_name_recognition",file_title=file_title,context=context)
    system_prompt=load_prompt('product_recognition_system')
    #2.加载模型  默认是千问
    llm=get_llm_client(json_mode=False)
    #3. 调用模型
    message=[{
        'role':'user','content':human_prompt},
        {'role': 'system', 'content': system_prompt}
    ]
    response=llm.invoke(message)
    item_name=response.content
    if not item_name:
        item_name=file_title

    return item_name






def step_4_chunks_and_state(state, chunks, item_name):
    """
    将item_name存入state,以及chunks中，并且把chunks在state中更新
    :param state:
    :param chunks:
    :param item_name:
    :return:
    """
    state['item_name']=item_name
    for chunk in chunks:
        chunk['item_name']=item_name
    state['chunks']=chunks

    logger.info(f"完成了对item_name及相关状态的更新")


def step_5_generate_embedding(item_name):
    """
    根据item_name生成向量
    :param item_name:
    :return:   dense_vector,sparse_vector
    """
    #主要通过函数.encode_documents(texts)生成结果  texts是一个字符串列表
    #结果：result={"dense":[,,,],
    #            "sparse":[,,,]   }
    result=generate_embeddings([item_name])#[]虽然这里是个列表，但item_name通常只有一个，所以下面取0

    dense_vector,sparse_vector=result["dense"][0],result["sparse"][0]
    sparse_vector = normalize_sparse_vector(sparse_vector)#标准化稀疏向量，milvus要的是dict[int:float]类型的稀疏向量
    return dense_vector,sparse_vector

def step_6_save_vector_to_db(file_title, item_name, sparse_vector, dense_vector):
    """
    将向量和对应的字段保存到向量数据库中
    :param file_title:
    :param item_name:
    :param sparse_vector:
    :param dense_vector:
    :return:
    """
    #1获取milvus的客户端
    milvus_client=get_milvus_client()
    #2.判断是否存在要的集合，不存在就创建集合（表）
    if not milvus_client.has_collection(collection_name=milvus_config.item_name_collection):
    #3.创建集合
        #3.1 创建集合对应的列的信息，用shcema来创建field（字段）
        schema=milvus_client.create_schema(auto_id=True,enable_dynamic_field=True)

        #3.2 add fields to schema
        schema.add_field(field_name="pk",datatype=DataType.INT64,is_primary=True,auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        #3.3配置索引
        index_params=milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",#给哪个词建立所引
            index_name="dense_vector_index",#索引名字
            index_type="HNSW",#向量查找所用的策略
            metric_type="COSINE",#相似度比较方法
            params={"M":16,#每个节点最多连接多少个邻居，M越大，图更密，索引越精准，召回率越高，但索引内存占比越高，建索引越慢，所以2不一定越大越好
                    "efConstruction":200#每次新插入一个向量，帮他找的候选邻居的数量。优缺点和M差不多
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
            index_type="SPARSE_INVERTED_INDEX",#倒排
            metric_type="IP",
            #DAAT_MAXSCORE，一种高性能 TopK 剪枝检索算法，提前估算，如果估算一个文档得分很低，且低于当前topk的最低分，那就直接跳过，更块，更省内存
            params={"inverted_index_algo": "DAAT_MAXSCORE"},  # Algorithm used for building and querying the index

        )
        # 3.4 创建集合
        milvus_client.create_collection(
            collection_name=milvus_config.item_name_collection,
            schema=schema,
            index_params=index_params
        )
        #4. 删除之前的item_name
    milvus_client.load_collection(collection_name=milvus_config.item_name_collection)
    milvus_client.delete(collection_name=milvus_config.item_name_collection,
                         filter=f"item_name=='{item_name}'")
        #5.插入数据，向集合中插入最新的数据和向量
    item=[{"file_title":file_title,
          "item_name":item_name,
          "dense_vector":dense_vector,
          "sparse_vector":sparse_vector

    }]
    milvus_client.insert(collection_name=milvus_config.item_name_collection,data=item)
    logger.info(f"保存了item_name：{item_name}的数据到向量数据库中。")

def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    【核心节点】商品主体名称识别（node_item_name_recognition）
    整体流程：提取输入→构建上下文→大模型识别→回填数据→生成向量→存入Milvus
    核心目的：利用大模型从文档切片中精准识别商品/主体名称，并生成双路向量（稠密+稀疏）存入数据库
    后续扩展点：支持多主体识别、增加商品属性提取、对接其他向量库等
    :param state: 项目状态字典（ImportGraphState），必须包含chunks/file_title/task_id
    :return: 更新后的状态字典，新增item_name键，且chunks列表中每个元素新增item_name字段
    """
    # 初始化当前节点信息，用于任务监控和日志溯源
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行核心节点：【文档实体识别】{node_name}")
    # 将当前节点加入运行中任务，更新全局任务状态
    add_running_task(state["task_id"], node_name)

    try:
        #1.校验状态，file_title保底作为item_name
        chunks,file_title = step_1_get_chunks(state)

        #2.构建上下文环境，chunks选取top5个，拼接成context文本
        context = step_2_bulid_context(chunks)

        #3.调用模型，拼接提示词，识别chunks对应的item_name
        item_name = step_3_call_llm(context,file_title)

        #4.将item_name放入state中的chunks中，更新state
        step_4_chunks_and_state(state,chunks,item_name)

        #5.调用嵌入模型，将item_name转成向量
        dense_vector,sparse_vector = step_5_generate_embedding(item_name)

        #6.将向量存入kb_item_name数据库（id,file_title,item_name,稀疏，密集）
        step_6_save_vector_to_db(file_title,item_name,sparse_vector,dense_vector)



    except Exception as e:
        # 全局异常捕获：保证节点执行失败不崩溃整个流程，记录详细错误日志便于排查
        logger.error(f">>> 核心节点执行失败：【文档实体识别】{node_name}，错误信息：{str(e)}", exc_info=True)

    # 返回更新后的状态字典，传递Chunk结果到下游节点
    return state



# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")



    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()