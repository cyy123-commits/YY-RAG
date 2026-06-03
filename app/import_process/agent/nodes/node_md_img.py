import os
import re
import sys
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

from inscriptis.model.canvas import prefix
# MinIO相关依赖
from minio import Minio
from minio.deleteobjects import DeleteObject


# 【核心改造1：移除原生OpenAI，导入LangChain工具类和多模态消息模块】
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.format_utils import format_state
from app.utils.task_utils import add_running_task, add_done_task
# LLM客户端工具类（核心复用，替换原生OpenAI调用）
from app.lm.lm_utils import get_llm_client
# LangChain多模态依赖（消息构造+异常捕获）
from langchain.messages import HumanMessage
from langchain_core.exceptions import LangChainException
# 项目配置
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# 项目日志工具（统一使用）
from app.core.logger import logger
# api访问限速工具
from app.utils.rate_limit_utils import apply_api_rate_limit
# 提示词加载工具
from app.core.load_prompt import load_prompt

def step_1_get_content(state:ImportGraphState)->Tuple[str,Path,Path]:

    """
    从状态图中提取，初始化MD处理所需核心数据

    :param state: 流程的全局状态对象
    :return: 三元组(MD文件内容, MD文件路径对象, 图片文件夹路径对象)
    :raise FileNotFoundError: 当状态中无有效MD文件路径时抛出
    """
    md_file_path=state["md_path"]

    if not md_file_path:
        raise FileNotFoundError(f"全局状态中无有效MD文件路径：{state['md_path']}")

    path_obj=Path(md_file_path)

    #状态中有content就使用，无就从文件读取
    if not state["md_content"]:
        with open(md_file_path,"r",encoding="utf-8") as f:
            md_content=f.read()
    else:
        md_content = state["md_content"]
        logger.debug(f"从文件读取MD内容完成，文件大小：{len(md_content)} 字符")

    #获取图片文件夹的目录images，经过minerU解析出的md文件都会有一个images目录存放图片
    images_dir_obj=path_obj.parent / "images"

    return md_content,path_obj,images_dir_obj

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
def is_supported_image(filename:str)->bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    #分离文件名和后缀，选取小写后缀判断是不是在格式集合内
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

def find_image_in_md(md_content:str,image_filename:str,context_len:int = 100):
    """
    查找MD内容中指定图片的所有引用位置，并返回每个位置的上下文文本
    :param md_content: MD文件完整内容
    :param image_filename: 图片文件名（含后缀）
    :param context_len: 上下文截取长度，默认前后各100字符
    :return: 上下文列表，每个元素为(上文, 下文)元组，无匹配则返回空列表
    """

    #利用正则表达式查找图片上下文,需要注意的是即使是同一个图，他们位于md中不同位置，uudi也不同，就是说一个md中无重复
    pattern=re.compile(r"!\[.*?\]\(.*?"+re.escape(image_filename)+r".*?\)")
    content=None
    items=list(pattern.finditer(md_content))
    if not items:
        return None
    if item := items[0]:#这里finditer返回一个迭代器对象，所以需要转成list来取图片第一次出现位置
        start,end=item.span()
        pre_text=md_content[max(0,start-context_len):start]
        post_text=md_content[end:min(len(md_content),end+context_len)]
        #打印图片上下文
        logger.debug(f"图片[{image_filename}]匹配到引用，上文：{pre_text.strip()}")
        logger.debug(f"图片[{image_filename}]匹配到引用，下文：{post_text.strip()}")
        content=(pre_text,post_text)
        if content:
            logger.info(f"图片{image_filename},在{md_content[:100]}中，截取到上下文：{content}")

        return content


def step_2_scan_images(md_content:str,images_dir:Path)->List[Tuple[str,str,Tuple[str,str]]]:
    """
    扫描图片文件夹，过滤出「支持格式+MD中实际引用」的图片，组装处理元数据
    :param md_content: md文件内容
    :param images_dir: 图片文件路径对象
    :return: 待处理图片列表，每个元素为(图片文件名, 图片完整路径, 图片上下文)元组
    """
    targets=[]
    #遍历所有图篇
    for image_file in os.listdir(images_dir):
        #筛选掉不支持的格式的图片
        if not is_supported_image(image_file):
            logger.debug(f"图片格式不支持，跳过：{image_file}")
            continue
        #组装图片完整路径
        img_path=str(images_dir / image_file)
        #查找图片在md中的上下文
        context=find_image_in_md(md_content,image_file)
        if not context:
            logger.warning(f"图片未在MD中引用，跳过处理：{image_file}")
            continue

        #都得到了，组装成列表吧
        targets.append((image_file,img_path,context))#加入一个md重复引用一个图，取第一个上下文
        logger.info(f"图片加入待处理列表：{image_file}")
        logger.info(f"图片扫描完成，共筛选出待处理图片：{len(targets)} 张")
    return targets


def step_3generate_img_summaries(targets,stem):
    """
    利用视觉模型获取图片的内容描述

    :param targets: [（图片名，图片地址，(上文，下文)）]
    :param stem: md文件的名字
    :return: {图片名：图片描述,图片名：图片描述}
    """

    summaries={}
    #循环每一条targets数据，向视觉模型中输送，获取总结结果，存入字典
    #定义一个循环队列，定义一个队列对象就行，如果多个，我们这边限速函数可能没限速，但大模型那边限速
    request_times=deque()
    for image_file,image_path,context in targets:
        #1.官方api会有访问次数限制，所以这里要限速，使用“滑动窗口限流”，这里使用的模型一分钟10次
        apply_api_rate_limit(request_times,max_requests=200)
        #2.向模型中发起请求
        #2.1模型对象
        vm_model=get_llm_client(model=lm_config.lv_model)
        prompt=load_prompt(name="image_summary",root_folder=stem,image_content=context)

        #2.2 import base64   base64可以将二进制图片转变成文本2字符串，让模型能够接收
        with open(image_path,"rb") as f:
            image_base64=base64.b64encode(f.read()).decode("utf-8")#先编码塞进prompt，模型再解码还原图片，这样视觉模型就可以发挥了
        messages=[
            {
                "role":"user",
                "content":[
                    {    #图片
                        "type":"image_url",
                        "image_url":{
                            "url":f"data:image/jpeg;base64,{image_base64}"

                        },
                    },
                    #文本提示词
                    {"type":"text","text":f"{prompt}"}
                ]
            }
        ]
        #2.3 执行获取总结
        response=vm_model.invoke(messages)
        summary=response.content.strip().replace("\n","")
        summaries[image_file]=summary
        logger.info(f"图片{image_file}，总结结果：{summary}")
    logger.info(f"总结图片，获取结果：{summaries}")
    return summaries


def step_4_upload_images_and_replace_md_content(summaries, targets, md_content, stem):
    """
    将图片传递给minio服务器
    替换md中的图片和描述
    :param summaries: 图片名：描述
    :param targets:（图片名，图片原地址，（上文，下文））
    :param md_content:原md内容
    :param stem:md文件名   传文件名作用就是在minip中的桶中的upload_file目录下生成同名文件夹，存放images
    :return:新md
    """
    #minio存储结果：桶/upload-images/md文件夹名/图片.jpg
    minio_client=get_minio_client()
    #1.删除minio存储结果，同一个存储照片文件夹需要清空再修改

    #1.1获取要删除的文件，注意:env文件中村的minio_img_dir是带前面/的
    #获取指定桶指定对象下   桶/upload-images/对象名（这里是stem）
    object_list=minio_client.list_objects(minio_config.bucket_name,
                                          prefix=f"{minio_config.minio_img_dir[1:]}/{stem}",
                                          recursive=True )
    #创建删除对象的名，乌龟的屁股，minio文档
    delete_list=[DeleteObject(obj.object_name) for obj in object_list]
    #1.2 调用方法进行删除
    errors=minio_client.remove_objects(minio_config.bucket_name,delete_list)
    for error in errors:
        logger.error(f"删除对象失败：{errors}")
    logger.info(f"完成对{stem}对象中的图片清空，删除了{len(delete_list)}个图片对象")

    #2,将图片上传到minio
    #声明记录图片上传结果的字典,里面放图片名和网络地址
    images_url={}
    # URL 编码 stem 中的特殊字符（括号、空格、中文等），避免 markdown 图片 URL 被截断
    from urllib.parse import quote
    safe_stem = quote(stem, safe='')
    for image_file,image_path, _ in targets:
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=f"{minio_config.minio_img_dir}/{safe_stem}/{image_file}",
                file_path=image_path,
                content_type="image/jpeg"
            )
            #上传完之后记录
            images_url[image_file]=f"http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{safe_stem}/{image_file}"
            logger.info(f"完成对图片{image_file}的上传，访问地址为{images_url[image_file]}")
        except Exception as e:
            logger.error(f"上传图片{image_file}失败，失败原因：str{e}")

    #3.md中将图片替换得到新的md、
    #3.1将summaries与新的网络地址汇总，{图片名：（描述，地址）}
    image_info={}
    for image_file,summary in summaries.items():
        if url := images_url.get(image_file):
            image_info[image_file]=(summary,url)
    logger.info(f"图片处理汇总结果为：{image_info}")

    #3.2如果进行了修改，通过正则匹配到旧md中的图片内容，然后进行替换
    if image_info:
        """
        xxxx
        xxxxxxxx ![xx](图片地址/image_file) ->![summary](minio中得到的url)
        xxxxxxxxxxx
        xxxx
        xxx
        """
        for image_file,(summary,url) in image_info.items():
            rep=re.compile(r"!\[.*?\]\(.*?"+image_file+r".*?\)")
            md_content=rep.sub(f"![{summary}]({url})",md_content)
        logger.info(f"完成了对md内容的转换，新的内容为{md_content[:100]}")
    return md_content


def step_5_replace_md_and_save(new_md_content, md_path_obj):
    """
    对新的md——content进行配分，并且返回新地址
    :param new_md_content: 新内容
    :param md_path_obj: 老地址
    :return: 新地址
    """
    new_md_path_str=os.path.splitext(md_path_obj)[0]+"_new.md"
    with open(new_md_path_str,"w",encoding="utf-8") as f:
        f.write(new_md_content)
    logger.info(f"完成了新内容的写入，新的地址是{new_md_path_str}")
    return new_md_path_str


def node_md_img(state:ImportGraphState)->ImportGraphState:
    """
    MD文件的图片的处理节点，通过视觉模型理解图片并更新state，一共五步
    ————
    1。初始化读入md的内容，文件路径以及images路径，校验存在性，注意上个节点会有两种不同情况 ——》返回md_content,path_obj,image_dir
    2。扫描图片文件夹，选出md中真正引用的图片，借用正则表达式提取图片上下文——》返回[(图片名，图片路径，（上文，下文）)]
    3..进行图片内容的总结，结合上面得到的上下文
    4.输入

    :param state: 导入流程全局状态对象，包含task_id、md_path、md_content等核心参数
    :return: 更新后的全局状态对象（md_content/md_path为处理后新值）
    """

    # 动态获取函数名避免硬编码
    func_name = sys._getframe().f_code.co_name

    # 节点启动日志，打印当前工作流状态
    logger.debug(f"【{func_name}】节点启动，\n当前工作流状态：{format_state(state)}")

    # 开始：记录节点运行状态
    add_running_task(state["task_id"], func_name)


#步骤1，初始化数据，校验md文件内容以及md路径以及图片路径
    md_content,md_path_obj,images_dir_obj=step_1_get_content(state)
    #参数state中的md_path，md_content
    #响应 校验后的md_content，md路径对象，获取图片的文件夹images

    #这里还赋值以下内容是有可能第一个节点输出的是md，不是pdf，不经过第二个节点，conten没内容
    state["md_content"]=md_content

    #判断有无image_dir,没有这把就结束
    if not images_dir_obj.exists():
        logger.info(f"图片文件夹不存在，跳过图片处理：{images_dir_obj.absolute()}")
        return state


#步骤2，循环遍历images中的图片，筛选出在md中的然后提取上下文
    targets=step_2_scan_images(md_content,images_dir_obj)
    #参数：md——content，images_dir
    #响应：[(图片名，图片路径，（上文，下文）)]

#步骤3.进行图片内容的总结，结合上面得到的上下文
    #参数：1.第二步得到的targets，2，md文件名
    #响应：{图片名：总结，。。。}、
    summaries=step_3generate_img_summaries(targets,md_path_obj.stem)
#步骤4 上传图片到minio同时替换md中的图片（描述+url地址·）
    #响应：new_md_content
    new_md_content=step_4_upload_images_and_replace_md_content(summaries,targets,md_content,md_path_obj.stem)

#新的md内容保存，修改状态
    #参数 new_md_content,md_oath_obj
    #响应：new_md_path
    new_md_file_path=step_5_replace_md_and_save(new_md_content,md_path_obj)


    #最后更新状态
    state["md_content"]=new_md_content
    state["md_path"]=new_md_file_path


    # 节点结束日志，打印当前工作流状态
    logger.info(f"【{func_name}】节点结束，\n当前工作流状态：{state}")

    # 开始：记录节点运行状态
    add_done_task(state["task_id"], func_name)


    return state




if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
