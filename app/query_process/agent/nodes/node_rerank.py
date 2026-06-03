

from app.core.logger import logger
from dotenv import load_dotenv
import sys
from app.lm.reranker_utils import get_reranker_model


from app.utils.task_utils import add_running_task, add_done_task


def step_1_merge_chunks(state):

  """
  将不同源的数据塞到一个大列表中


  :param state:
  :return:
  """
  #1.准备数据
  rrf_chunks=state.get("rrf_chunks")
  web_docs=state.get("web_search_docs")

  #2.循环添加数据进入大列表
  chunks_list=[]
  #2.1 循环遍历rrf_chunks
  for chunk in rrf_chunks:
    entity=chunk.get("entity")
    chunk_id=entity.get("chunk_id")
    content=entity.get("content")
    title=entity.get("title")
    chunks_list.append({
      "chunk_id":chunk_id,
      "text":content,
      "title":title,
      "source":"local",
      "url":""
    })
  #2.2 循环遍历web_docs
  for chunk in web_docs:
    text=chunk.get("snippet")
    title=chunk.get("title")
    url=chunk.get("url")
    chunks_list.append({
      "chunk_id":"",
      "text": text,
      "title": title,
      "source": "web",
      "url": url

    })

  #返回结果
  logger.info(f"{len(chunks_list)} chunks merged")

  return chunks_list


def step_2_rerank_chunks_list(chunks_list, state):
  """
  用rerank进行精排
  :param chunks_list:
  :param state:这里需要使用state中的问题来加上chunks中的回答内容进行打分，因此需要state
  :return:
  """
  #1.获取问题
  rewritten_query=state.get("rewritten_query")
  #2.组合问题和回答
  #获取回答列表
  text_list=[chunk["text"] for chunk in chunks_list]
  #构建输入rerank模型的参数，[(问题，答案)]
  input_pairs=[[rewritten_query,text] for text in text_list]
  #初始化rerank模型
  rerank=get_reranker_model()

  if not chunks_list:  # 添加空列表检查
    logger.warning("chunks_list为空，跳过重排序")
    return []

  #获取得分
  score=rerank.compute_score(input_pairs,normalize=True)#类似归一化，缩放使得分差变小，为了后面断崖算法好比较


  #追加score
  chunks_list_with_score=[]

  for score,item in zip(score,chunks_list):
    item["score"]=score
    chunks_list_with_score.append(item)

  chunks_list_with_score.sort(key=lambda x: x["score"],reverse=True)

  logger.info(f"{len(chunks_list_with_score)} chunks reranked,完成排序和打分")

  return chunks_list_with_score





# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 1
# 断崖阈值（相对）
RERANK_GAP_RATIO: float = 0.25
# 断崖阈值（绝对）
RERANK_GAP_ABS: float = 0.5


def step_3_topk_and_gap(rerank_score_list):
  """
  通过双指针方式从左往右（因为已经根据score排好序）两两对比score，进行topk的防截断处理


  :param rerank_score_list:
  :return:
  """
  #参数
  topk=RERANK_MAX_TOPK
  min_topk=RERANK_MIN_TOPK
  abs_gap=RERANK_GAP_ABS
  rela_gap=RERANK_GAP_RATIO

  #注意：topk不能大于rerank_score_list的长度
  topk=min(len(rerank_score_list),topk)
  #1.循环数据列表，进行双指针的处理与比较
  if topk>min_topk:
  #1.1正常循环，topk>min_topk正常双指针循环,循环最小topk到topk
    for index in range(min_topk-1,topk-1):
      score1=rerank_score_list[index].get("score")
      score2=rerank_score_list[index+1].get("score")
      gap=score1-score2

      rela=gap/(abs(score1)+1e-8)
      if gap>=abs_gap or rela>=rela_gap:
        logger.info(f"数据集合{rerank_score_list}在索引为{index}处发生了断崖")
        topk=index+1
        break
  #1.2 topk=min_tok or topk<min_tok,直接去topk就行了
  #2.截断确定的数量topk
  topk_docs_list=rerank_score_list[:topk]


  #3.返回截断的数据列表
  return topk_docs_list




def node_rerank(state):
  """
  节点作用：把rrf+mcp-->进行rank精排-》打分-》算法-》topk

  入参：主要用的是chunks,和web_documents进行融合排名，使用rerank模型进行打分重排
  chunks:[{id: ,distance:, entity:{chunk_id: ,content:,title:,item_name:,.....}},{},{}....]
  web_documents:[{snippet:内容，title:标题，url:关联的文章或者图片的地址}，{},{},{},{}]

  算法分析：最多存前十，最少返回1个
          绝对阈值就是说前者比后者大0.5以上，那就直接截断，不往下取了
          相对阈值就直接把前者与后者的差距/前者，如果大于0.25就截取，防止出现0.8，0.7，0.6.。这类一直不触发绝对阈值的情况
  :param state
  :return:  返回一堆
  """
  node_name=sys._getframe().f_code.co_name
  logger.info(f"{node_name}节点开始执行")
  add_running_task(state["session_id"],node_name,state["is_stream"])


  #加载数据
  #1.非同源数据结果合并，放到一个集合中
  """
  rrf=[{id: ,distance:, entity:{chunk_id: ,content:,title:,item_name:,.....}},{},{}....]
  mcp_docs=[{snippet:内容，title:标题，url:关联的文章或者图片的地址}，{},{},{},{}]
  把他们合并提取出结果：
  [
    {
    text:content/snippet,
    chunk_id:chunk_id,(rrf源有）
    title:rrf/mcp,
    url:mcp有，rrf无
    source:local/web   这里定义rrf来的源为local，，网络搜索来的源为web
    }
  ]
  """
  chunks_list=step_1_merge_chunks(state)

  #2.调用rerank模型进行打分，并把score回填到chunks_list

  rerank_score_list=step_2_rerank_chunks_list(chunks_list,state)
  """
  [
    {
    text:content/snippet,
    chunk_id:chunk_id,(rrf源有）
    title:rrf/mcp,
    url:mcp有，rrf无
    source:local/web   这里定义rrf来的源为local，，网络搜索来的源为web
    score:score
    }
  ]
  """
  #3.使用算法进行topk防断崖处理
  final_doc_list=step_3_topk_and_gap(rerank_score_list)


  #4.结果放入state
  state["reranked_docs"]=final_doc_list

  logger.info(f"{node_name}节点执行完成，截取出{len(final_doc_list)}个结果")
  add_done_task(state["session_id"],node_name,state["is_stream"])
  return state





if __name__ == "__main__":
  print("\n" + "=" * 50)
  print(">>> 启动 node_rerank 本地测试")
  print("=" * 50)

  # 1. 模拟数据
  # 1.1 RRF 本地文档数据
  mock_rrf_chunks = [
    {"entity":{"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍", "score": 0.9}},
    {"entity":{"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍", "score": 0.8}},
    {"entity":{"chunk_id": "local_3", "content": "无关的测试文档内容", "title": "测试文档", "score": 0.1}} # 预期低分
  ]

  # 1.2 MCP 联网搜索数据
  mock_web_docs = [
    {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
    {"title": "无关网页", "url": "http://web.com/2", "snippet": "今天天气不错，适合出去游玩"}  # 预期低分
  ]

  mock_state = {
    "session_id": "test_rerank_session",
    "rewritten_query": "什么是RRF和Rerank？",  # 查询意图：想了解这两个算法
    "rrf_chunks": mock_rrf_chunks,
    "web_search_docs": mock_web_docs,
    "is_stream": False
  }

  try:
    # 运行节点
    result = node_rerank(mock_state)
    reranked = result.get("reranked_docs", [])

    print("\n" + "=" * 50)
    print(">>> 测试结果摘要:")
    print(f"输入文档总数: {len(mock_rrf_chunks) + len(mock_web_docs)}")
    print(f"输出文档总数: {len(reranked)}")
    print("-" * 30)

    print("最终排名:")
    for i, doc in enumerate(reranked, 1):
      print(f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:20]}...")

    # 验证逻辑：
    # 预期 "local_1", "local_2", "Rerank技术详解" 分数较高
    # 预期 "local_3", "无关网页" 分数较低，可能被截断或排在最后

    top1_score = reranked[0].get("score")
    if top1_score > 0:
      print("\n[PASS] Rerank 打分正常")
    else:
      print("\n[FAIL] Rerank 打分异常 (均为0或负数)")

    print("=" * 50)

  except Exception as e:
    logger.exception(f"测试运行期间发生未捕获异常: {e}")






