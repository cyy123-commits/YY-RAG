import sys



from app.utils.task_utils import add_running_task, add_done_task, set_task_result,update_task_status
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
import re

_IMAGE_BLOCK_MARKER = "【图片】"
MAX_CONTEXT_CHARS = 12000


def step_1_check_answer(state):
    #判断有没有answer
    #1.获取answer 以及is_stream
    answer=state.get("answer")
    is_stream=state.get("is_stream")
    #1.1是否有aswer
    if answer:
        #1.1有answer，判断是否是流式，
        #流式
        if is_stream:
        #随送到sse
            push_to_session(state['session_id'],SSEEvent.DELTA,{"delta":answer})

        #非流式
        #设置任务结果set_task_result
        else:
            set_task_result(state['session_id'],"answer",answer)
        #返回True
        return True
    #无answer，返回False
    else:
        return False


def step_2_load_prompt(state):
    """
    把状态中的reranked_docs,history,item_names，rewritten_query等问题拼进提示词

    :param state:
    :return:
    """

    #获取state中的内容
    rewritten_query=state.get("rewritten_query") or state.get("original_query")
    reranked_docs=state.get("reranked_docs",[])
    item_names=state.get("item_names",[])
    history=state.get("history",[])
    #1。处理final_text,将rerank节点得到的reranked_docs按照一定形式进行处理
    """
    [
        [1][text][source][title][score]\n\n
        [2][text][source][title][score]\n\n
    ]
    """
    docs=[]
    text_length=0
    for index,chunk in enumerate(reranked_docs,start=1):
        text=chunk.get("text")
        title=chunk.get("title")
        source=chunk.get("source")
        score=chunk.get("score")

        content=f"[{index}][source={source}][title={title}[score={score}]\n\n{text}"
        text_length+=len(content)
        if text_length>MAX_CONTEXT_CHARS:
            logger.info(f"本次内容追加长度过大，停止追加")
            break
        docs.append(content)
    final_text='\n\n'.join(docs)

    #2.处理history
    history_str=""
    if history and len(history)>0:
        for i,message in enumerate(history,start=1):
            role=message.get("role")
            text=message.get("text")
            current_history=""
            if role=="user" and text:
                current_history+=f"【用户】：{text}"
            elif role=="assistant" and text:
                current_history+=f"【助手】：{text}"
            text_length+=len(current_history)
            if text_length>MAX_CONTEXT_CHARS:
                logger.info(f"本次内容追加长度过大，停止追加")
                break
            history_str+=current_history
    else:
        history_str="没有历史对话记录"

    #3.处理item_names
    item_names_str="，".join(item_names)

    #4.处理最终question提示词,这个模型生成提示词的提示词中明确说了不要翻译文本中的图片，所以这里产生的最终提示词就没有图片
    answer_out_prompt=load_prompt("answer_out",context=final_text,
                                  history=history_str,
                                  item_names=item_names_str,
                                  question=rewritten_query)
    logger.info(f"已经完成提示词生成")
    return answer_out_prompt


def step_3_create_answer(state, answer_out_prompt):
    """
    使用模型生成最终答案

    :param state:
    :param answer_out_prompt:
    :return:
    """
    #1获取模型对象和客户端
    model= get_llm_client()
    answer=''
    #2.获取流式状态【sse||set_result】
    is_stream=state.get("is_stream",False)
    if is_stream:
        # 3，调用模型  sse 用 stream    非流式 用 invoke
        for chunk in model.stream(answer_out_prompt):
            delta=chunk.content
            answer+=delta
            push_to_session(state['session_id'],SSEEvent.DELTA,{"delta":delta})

    else:
        response=model.invoke(answer_out_prompt)
        content=response.content
        answer=content
        set_task_result(state['session_id'],"answer",answer)



    #4，最终答案赋值给state
    state['answer']=answer

    #5.返回结果answer
    logger.info(f"模型返回最终结果:{answer}")
    return answer


def step_4_extract_image_url(state):
    """
    从 topk_list 中每个 doc 提取图片 URL
    支持三种格式：1) markdown ![alt](url)  2) doc.url 字段  3) 纯 URL（http...jpg/png等）
    :param state:
    :return:
    """
    images=[]
    set_images=set()

    # 正则1: markdown 图片语法 ![alt](url)
    # 使用 .+? 配合 (?:\s|$) 确保正确匹配 URL 中包含括号的情况
    # 例如: ![desc](http://url/foo(bar)/img.jpg) — 不会在 bar) 处截断
    md_image_reg = re.compile(r"!\[.*?\]\((.+?)\)(?:\s|$)", re.MULTILINE)
    # 正则2: 纯图片 URL（http/https 开头，以图片扩展名结尾）
    plain_url_reg = re.compile(r"https?://[^\s]+?\.(?:png|jpe?g|gif|webp|svg|bmp)", re.IGNORECASE)

    reranked_docs = state.get("reranked_docs", [])
    for doc in reranked_docs:
        # 方式1: doc.url 字段（网络搜索结果）
        url = doc.get("url")
        if url:
            if url.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                if url not in set_images:
                    images.append(url)
                    set_images.add(url)

        # 方式2 + 3: 从 chunk text 中提取
        text = doc.get("text")
        if text:
            # markdown 图片语法
            for image_url in md_image_reg.findall(text):
                if image_url not in set_images:
                    images.append(image_url)
                    set_images.add(image_url)
            # 纯 URL 兜底（有些 chunk 可能直接写 http://...jpg）
            for image_url in plain_url_reg.findall(text):
                if image_url not in set_images:
                    images.append(image_url)
                    set_images.add(image_url)

    logger.info(f"已经完成图片提取，数量：{len(images)}, 提取内容：{images}")
    state["image_urls"] = images
    return images


def step_5_write_history(state):
    """
    将对话记录存储到mongodb
    每次对话对应两条history
    user query->text
    assiant  asnwer->text

    :param state:
    :return:
    """
    session_id=state.get("session_id")
    answer=state.get("answer")
    rewritten_query=state.get("rewritten_query")
    item_names=state.get("item_names")

    #前面保存了用户的,这里只存一个回答就行
    if answer:
        save_chat_message(
            session_id=session_id,
            role="assistant",
            text=answer,
            item_names=item_names,
            rewritten_query=rewritten_query
        )
    logger.info("已经将本次对话记录的保存到数据库")



def node_answer_output(state: QueryGraphState) -> QueryGraphState:
    """
    实现大概流程： topk的docs->大模型->根据提示词生成答案->流式用sse-》前端（push_to_session） || 如果非流式：get_task_result
    #1,先检查state中有没有answer，因为第一个实体识别节点可能没有确定的item_name之后就会生成answer
    #2,生成对应的提示词prompt
    #3.使用模型润色答案 ->结果 -》文本,这里只要文本，会跟模型说
    #4，提取topklist中的图片地址，单独返回，用sse
    #5.存储对话聊天记录（user/assiant），这里主要存储assiant的记录，user的问题记录村过了
    #6.sse-final事件->返回图片
    :param state:
    :return:
    """
    print("---node_answer_output节点处理开始-----")
    node_name=sys._getframe().f_code.co_name
    add_running_task(state['session_id'],node_name,state.get("is_stream"))


    #1.检查state中是否存在answer,有就直接返回结果了
    answer_exist=step_1_check_answer(state)

    if answer_exist:
        # 已有预置 answer（如 node_item_name_confirm 无确认 item_name 时），直接从 state 取值
        answer = state.get("answer", "")
    else:
        # 2.没有 生成提示词
        answer_out_prompt=step_2_load_prompt(state)

        #3，没有 使用模型润色答案-》结果-》文本回答
        answer=step_3_create_answer(state,answer_out_prompt)

    #4.提取topklist中的图片地址（无论走哪条分支都要执行，确保图片不丢失）
    images_urls=step_4_extract_image_url(state)



    #6.添加聊天记录（mongodb）
    step_5_write_history(state)


    add_done_task(state['session_id'],node_name,state.get("is_stream"))
    update_task_status(state['session_id'],"completed",state.get("is_stream"))

    #5.sse-final事件 → 无论有无图片都必须发送，否则前端一直等待
    # 注意：done_task 必须在 FINAL 之前调用，否则前端收到 FINAL 后关闭连接，
    # 导致 done_task 推送的 progress 事件丢失，前端一直显示"处理中"和⏳
    push_to_session(state["session_id"], SSEEvent.FINAL,
                    {"answer": answer,
                     "status": "completed",
                     "image_urls": images_urls or []
                     }
                    )
    print("---node_answer_output节点处理结束-----")

    return state





