import sys
import os
from typing import Any, List, Dict

from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.utils.task_utils import add_running_task,add_done_task
from app.core.logger import logger


def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:

    # 获取当前节点名称，用于日志和任务状态记录
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行LangGraph节点：{current_node}")

    # 标记任务运行状态，用于任务监控/前端进度展示
    add_running_task(state.get("task_id", ""), current_node)
    logger.info("--- BGE-M3 文本向量化处理启动 ---")

    try:
        #1.获取要生成向量的chunks
        chunks=state.get('chunks')
        if not chunks or not isinstance(chunks,list):
            raise  ValueError(f"找不到chunks或者chunks类型不匹配")

        #2.将chunks生成向量
        #2.1获取嵌入式模型，客户端类的代码已经获取了，并且其中还有了generate_embedding生成向量的方法
        #2.2 批量生成向量
        """
        这里需要注意，
        1. 一定要计划好把chunks中的啥内容生成向量
        本次把item_name和content（里面就有title），这样能更精确
        并以f"商品名：item_name,介绍：content"的形式来进行向量化
        2. 生成向量要放列表嘛，这里要放合适大小的列表，比如这个m3他的上下文窗口有8192个token，根据自己的chunks内容大小来决定放几个字符串
        这里放5个
        """

        final_chunks=[]
        batch_size=5
        for i in range(0,len(chunks),batch_size):
            #本批次的chunks，得到当前批次的转向量内容列比奥
            batch_chunks=chunks[i:i+batch_size]
            current_text=[]
            for chunk in batch_chunks:
                item_name= chunk.get("item_name")
                content=chunk.get("content")
                chunk_text=f"商品名：{item_name},介绍：{content}"
                current_text.append(chunk_text)

            #生成当前批次的向量
            result=generate_embeddings(current_text)
            #将向量添加到chunk中
            # 完善chunk的属性，添加稀疏和稠密向量

            for i,chunk in enumerate(batch_chunks):
                chunk_item=chunk.copy()
                chunk_item['dense_vector']=result["dense"][i]
                chunk_item['sparse_vector']=result["sparse"][i]
                final_chunks.append(chunk_item)
            # 更新state
            state["chunks"]=final_chunks
            logger.info(f"BGE-m3完成了对chunk向量化的处理，共处理了{len(final_chunks)}个文本切片")


        add_done_task(state.get("task_id", ""), current_node)
    except Exception as e:
        # 捕获节点所有异常，记录错误堆栈，不中断整体流程
        logger.error(f"BGE-M3向量化节点执行失败：{str(e)}", exc_info=True)

    # 返回更新后的状态对象，传递至下游节点
    return state


