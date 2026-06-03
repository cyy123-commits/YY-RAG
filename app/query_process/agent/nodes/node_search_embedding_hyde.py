import sys
from app.clients import milvus_utils
from langchain_core.messages import HumanMessage
from app.clients.milvus_utils import create_hybrid_search_requests, hybrid_search
from app.core.load_prompt import load_prompt
from app.lm.embedding_utils import generate_embeddings
from app.core.logger import logger
from app.lm.lm_utils import get_llm_client
from app.utils.task_utils import add_running_task, add_done_task
from app.conf.milvus_config import milvus_config


def step_1_create_hyde_doc(rewritten_query):
   #初始化大模型客户端
   llm_client=get_llm_client()

   #构建提示词
   prompt=load_prompt("hyde_prompt",rewritten_query=rewritten_query)
   message=[
      HumanMessage(content=prompt,)
   ]

   #调用大模型
   response=llm_client.invoke(message)

   hyde_doc=response.content
   logger.info(f"使用模型生成假设性答案，问题：{rewritten_query}.答案：{hyde_doc}")
   return hyde_doc


def step_2_search_embedding_hyde(rewritten_query, hyde_doc, item_names):
   """
   根据问题+假设性答案进行向量检索
   :param rewritten_query:
   :param hyde_doc:
   :param item_names:
   :return:
   """
   #1.向量化
   text=rewritten_query+hyde_doc
   embedding_hyde=generate_embeddings([text])
   dense_vector=embedding_hyde["dense"][0]
   sparse_vector=embedding_hyde["sparse"][0]
   #构建AnnSearchRequest：构造 Milvus 过滤表达式 item_name in ["A", "B"]
   items_str = '["' + '", "'.join(item_names) + '"]'
   reqs=create_hybrid_search_requests(
      dense_vector=dense_vector,
      sparse_vector=sparse_vector,
      expr=f"item_name in {items_str}"
   )

   #混合向量检索
   milvus_client=milvus_utils.get_milvus_client()
   response=hybrid_search(
      client=milvus_client,
      collection_name=milvus_config.chunks_collection,
      reqs=reqs,
      ranker_weights=(0.7,0.3),
      norm_score=True,
      limit=5,
      output_fields=["chunk_id", "content", "title", "file_title", "parent_title", "item_name"]

   )

   #处理结果
   logger.info(f'')
   result=response[0] if response else []
   logger.info(f"假设性问题检索结果：{result}")
   return result


def node_search_embedding_hyde(state):
   """
   假设性答案：问题=》模型=》给一个假设下答案=》问题+答案=》搜索

   #节点功能：让LLM生成假设性答案吧，再通过答案进行向量检索，提高召回率
   :param state:
   :return:
   """

   print("hyde处理开始")
   add_running_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))

   #1，提取参数   item_names   rewritten_qiery
   item_names=state.get("item_names")
   rewritten_query=state.get("rewritten_query")
   #2。调用大模型，生成假设性回答
   hyde_doc=step_1_create_hyde_doc(rewritten_query)

   #3.把问题+答案进行向量化,进行混合检索
   result=step_2_search_embedding_hyde(rewritten_query,hyde_doc,item_names)


   #4.赋值返回结果
   add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

   print("hyde处理结束")
   return {"hyde_embedding_chunks":result}
