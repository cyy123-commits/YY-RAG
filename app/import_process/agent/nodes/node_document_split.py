import re
import json
import os
import sys
# 统一类型注解，避免混用any/Any
from typing import List, Dict, Any, Tuple
# LangChain文本分割器（标注核心用途，便于理解）
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.sql.operators import is_precedent

# 项目内部工具/状态/日志导入（保持原有路径）
from app.utils.task_utils import add_running_task
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger  # 项目统一日志工具，核心替换print

# --- 配置参数 (Configuration) ---
# 单个Chunk最大字符长度：超过则触发二次切分（适配大模型上下文窗口）
DEFAULT_MAX_CONTENT_LENGTH = 2000
# 短Chunk合并阈值：同父标题的短Chunk会被合并，减少碎片化
MIN_CONTENT_LENGTH = 500

def step_1_get_inputs(state: ImportGraphState) -> Tuple[Any, str, int]:
    """
    步骤1 获取并预处理输入数据
    功能：从状态字典中提取MD内容/文件标题/最大长度，做基础标准化
    :param state: 项目状态字典（ImportGraphState），包含md_content等核心键
    :return: 标准化后的MD内容/文件标题/单个Chunk最大长度（无内容则返回None,None,None）
    """
    # 从状态中提取MD原始内容
    md_content = state.get("md_content")
    # 空内容兜底：无MD内容则直接返回，终止后续处理
    if not md_content:
        logger.error("状态字典中无有效MD内容，终止文档切分")
        raise Exception(f"未找到文档内容，请检查文件路径是否正确")

    #处理md_content中的换行符号
    """
    不同系统中换行符有所不同，有\r\n,\r,\n三种，统一换成\n
    """
    content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    # 提取文件标题：有则用，无则默认"Default_file"
    file_title = state.get("file_title", "Default_file")
    # 提取最大Chunk长度：有则用状态中的配置，无则用全局默认值
    max_len = DEFAULT_MAX_CONTENT_LENGTH


    logger.info(f"步骤1：输入数据加载完成，文件标题：{file_title}，最大Chunk长度：{max_len}")
    return content, file_title, max_len

#粗切割
def step_2_split_by_titles(md_content: str, file_title: str) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    【步骤2】按Markdown标题初次切分（核心：按#分级切分，跳过代码块内标题）
    :param md_content: 标准化后的MD完整内容（字符串）
    :param file_title: 所属文件标题，用于标记章节归属
    :return: [{content.title,file_title},{},{}]

    比如md中的内容是，若要返回{content.title,file_title}，首先要判断是不是代码块，防止##注释被误判为标题，之后判断是否是标题
    ##标题
    内容\n
    ！[]()
    内容\n
    内容\n

    ##标题
    内容\n
    python代码块
    ##python注释
    内容\n
    内容\n

    ##标题
    内容\n
    内容\n
    内容\n
    """
    #1，前置准备工作，
    #1.1 正则
    title_pattern=r'^\s*#{1,6}\s+.+'
    #1.2 md_content切割 按照\n切割出lines，后面会循环遍历每行来判断
    lines=md_content.split("\n")
    #1.3 不着急往字典里存，临时存储，current_title:str current_lines=[]  title_count=0(存储了多少块)
    #                          is_code_block:bool=False 是不是代码块
    current_title="" #当前标题
    current_lines=[]  #当前标题下的行
    title_count=0
    is_code_block=False
    #1.4 最终存储的列表 sections=[]
    sections=[]

    #2. 循环每行的列表
    for line in lines:
        line=line.strip()
        # 2.1 判断是不是代码块
        #先判断是不是代码块,代码块在md中以```开头或者结尾
        if line.startswith('```') or line.endswith('```'):
            #代表进入或者即将走出代码块,但是第一次进来一定是走入代码块，所以is——code设置反
            is_code_block=not is_code_block
            current_lines.append(line)
            continue
        # 2.2 判断是不是标题
        #上面只是判断是否进入了代码块（遇到```），代码块内的详细内容还要进行判断
        #判断是不是标题
        is_title=not is_code_block and re.match(title_pattern,line)

        if is_title:
         # 2.3 是标题怎末处理,遍历到一个新的标题时，将旧标题，旧标题下的lines都存到sections中
            if current_title:
                sections.append({
                    "title":current_title,
                    "content":'\n'.join(current_lines),
                    "file_title":file_title
                })
            current_title=line
            current_lines=[current_title]
            title_count+=1


        #2.4 不是标题怎末处理
        else:
            current_lines.append(line)
    #当遍历完最后一个标题的内容时，就不会让is_title为ture了，所以不会触发存储，所以要再存储最后一个
    if current_title:
        sections.append({
            "title": current_title,
            "content": '\n'.join(current_lines),
            "file_title": file_title
        })
    #3. 返回结果
    logger.info(f"已经完成chunks的语义粗切，识别到的chunk数量：{title_count}")
    return sections,title_count,len(lines)





def _split_long_section(section: Dict[str, Any], max_length: int = DEFAULT_MAX_CONTENT_LENGTH) -> List[Dict[str, Any]]:
    """
    【辅助函数】超长章节二次切分（核心适配LangChain分割器）
    功能：单个章节内容超限时，按「段落→句子→空格」从粗到细切分，保留语义
    切分规则：1.先按空行(段落) 2.再按换行 3.最后按中英文标点/空格
    :param section: 原始章节字典，必须包含content键，可选title/file_title等
    :param max_length: 单个Chunk最大字符长度，默认使用全局配置
    :return: 切分后的子章节列表，每个子章节带父标题parent_title/序号part等元信息
    """
    #1.获取content
    content=section["content"]
    #2.判断content是否超长
    if len(content) <= max_length:
        logger.info(f"当前Chunk长度小于等于{max_length},不做二次切割")
        return [section]
    #3.超长了，进行二次切割
    spliter=RecursiveCharacterTextSplitter(
        chunk_size=max_length,#切割的最大长度
        chunk_overlap=100,#重叠长度
        separators=['\n\n','\n','。','!','，','：',' ']#顺序的切割，如果根据上一个符号切割大于最大长度1了，那就根据下一个再切
    )

    #切出来的section中要有title=标题名 _1,_2,_3...  part=1,2,3... parent_title=section.title
    sub_sections=[]
    for index,chunk in enumerate(spliter.split_text(content),start=1):
        text=chunk.strip()
        title=f"{section.get("title")}_{index}"
        part=index
        parent_title=section.get("title")
        file_title=section.get("file_title")
        sub_sections.append({
            "title": title,
            "content":text,
            "parent_title": parent_title,
            "part": part,
            "file_title":file_title

        })

    return  sub_sections









def _merge_short_sections(sections: List[Dict[str, Any]], min_length: int = MIN_CONTENT_LENGTH,maxlength:int=DEFAULT_MAX_CONTENT_LENGTH) -> List[Dict[str, Any]]:
    """
    上一次切的太碎，还需要合并
    核心规则：1.content长度小于min_len，2，合并的必须时同一个parent_title
    :param sections: 待合并的Chunk列表（通常是_split_long_section切分后的结果）
    :param min_length: 最小长度阈值，低于此值的Chunk会被合并
    :return: 合并后的Chunk列表，长度适中，保留元信息
    """
    merged_sections=[]
    pre_section=None
    for section in sections:
        #第一次来就直接放进来，先不用看其他条件
        if pre_section is None:
            pre_section=section
            continue
        #pre_section中有东西之后就要判断，当前pre的content长度是否小于最小值，以及pre与当前sec的parent——title是否一致
        is_pre_section_short=len(pre_section.get('content')) < min_length
        is_same_parent_title=section.get("parent_title") == pre_section.get("parent_title") and section.get("parent_title")

        if is_pre_section_short and is_same_parent_title:
            #上一次既是短块又与这次循环的section是一个父标题->合并内容
            # parent_title=section.get("parent_title")
            current_content=section.get("content")

            pre_section["content"]+="\n\n"+current_content

            pre_section["part"]=section.get("part")

        else:
            #上一次的不是短块，或者与本次不是同一个父标题，直接添加到列表
            merged_sections.append(pre_section)
            pre_section=section#这次的section就作为后面要融合别人的pre_section了

    #最后一个section被遍历完之后加入到了pre_section中，跳出循环，那pre_section并没有加入到merage中，所以要处理一下
    if pre_section is not None:
        merged_sections.append(pre_section)
    return merged_sections





#精细切割
def step_3_refine_chunks(sections: List[Dict[str, Any]], max_len: int,min_len: int) -> List[Dict[str, Any]]:
    """
    【步骤3】Chunk精细化处理（核心：长切短合，适配大模型/检索）
    执行流程：1.切分超长章节 2.合并过短章节 3.父标题兜底（适配Milvus向量库schema）
    :param sections: 步骤3处理后的章节列表
    :param max_len: 单个Chunk最大字符长度
    :param min_len: 单个Chunk最小字符长度
    :return: 长度适中、低碎片化的最终Chunk列表
    """

    final_sections=[] #存储处理后的块

    #超过的先切碎
    for section in sections:
        #[{title,content,file_title,parent_title,part},{},{}]
        sub_section=_split_long_section(section,max_len)
        final_sections.extend(sub_section)#平铺

    #小的再合并
    final_sections=_merge_short_sections(final_sections,min_len)
    #补全属性和参数 part parent_title
    for section in final_sections:
        section['part']=section.get('part') or 1
        section['parent_title']=section.get('parent_title') or section.get('title')


    #返回
    logger.info(f"完成了对chunk的精细切割")
    return final_sections




def step_4_backup(state: ImportGraphState, sections: List[Dict[str, Any]]):
    """
    将切割完的碎片进行存储
    :param state: 项目状态字典，需包含local_dir（备份目录）
    :param sections: 最终处理后的Chunk列表
    """
    local_dir=state.get('local_dir')
    backup_file_path=os.path.join(local_dir,"backup")
    with open(backup_file_path,"w",encoding="utf-8") as f:
        json.dump(
            sections,#将啥内容写到指定文件流
            f,#写到的位置
            ensure_ascii=False,#中文直接原文存储
            indent=4#json带有缩进4

        )
    logger.info(f"已经将内容进行备份，备份到：{backup_file_path}")


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    【核心节点】文档切分主节点（node_document_split）
    整体流程：加载输入→按MD标题初切→无标题兜底→长切短合→统计输出→结果备份
    核心目的：将长MD文档切分为长度适中的Chunk，适配大模型上下文窗口和向量检索
    后续扩展点：可在各步骤间新增Chunk元信息补充、自定义切分规则、向量入库前置处理等
    :param state: 项目状态字典（ImportGraphState），必须包含md_content/task_id；可选local_dir/max_content_length/file_title
    :return: 更新后的状态字典，新增chunks键（存储最终处理后的Chunk列表，每个Chunk为含title/content/parent_title的字典）
    """
    # 初始化当前节点信息，用于任务监控和日志溯源
    node_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行核心节点：【文档切分】{node_name}")
    # 将当前节点加入运行中任务，更新全局任务状态
    add_running_task(state["task_id"], node_name)

    try:
        # ===================================== 步骤1：加载并标准化输入数据 =====================================
        # 作用：从状态字典提取MD内容/文件标题/Chunk最大长度，统一换行符消除系统差异，做空值兜底
        # 输出：标准化后的md_content、文件标题、单个Chunk最大长度；无有效MD内容则直接终止节点执行
        md_content, file_title, max_len= step_1_get_inputs(state)
        if md_content is None:
            logger.info(f">>> 节点执行终止：{node_name}（无有效MD内容）")
            return state

        # ===================================== 步骤2：粗切割=====================================
        # 作用：基于markdown内容的标题进行切割，保证语义，粗切
        # 输出：[{content:标题下的内容,title:标题,file_name:文件名}，{},{}]
        sections, title_count, lines_count = step_2_split_by_titles(md_content, file_title)

        # =====================================3.无标题场景兜底处理 =====================================
        # 作用：解决MD文档无任何标题的边界情况，避免后续切分逻辑异常
        # 输出：有标题则返回步骤2的章节列表；无标题则将全文封装为单个「无标题」章节，保证数据格式统一
        if title_count == 0:
            sections=[{
                "title":"无标题",
                "content":md_content,
                "file_title":file_title
            }]
        # ===================================== 步骤4：Chunk精细化处理（长切短合） =====================================
        # 作用：核心切分逻辑，先将超长章节按「段落→句子」二次切分，再合并同父标题的过短章节，减少碎片化
        # 额外处理：对所有Chunk做parent_title兜底，适配Milvus向量库必填字段要求
        # 输出：长度适中、语义完整、低碎片化的最终Chunk列表（可直接用于向量入库/大模型调用）
        sections = step_3_refine_chunks(sections,max_len, MIN_CONTENT_LENGTH,)


        # ===================================== 步骤5：Chunk结果本地JSON备份 + 状态更新 =====================================
        # 作用1：将最终Chunk列表备份到local_dir目录的chunks.json，便于后续问题排查、数据复用
        # 作用2：将Chunk列表写入状态字典，传递给下一个节点（如向量入库、大模型摘要等）
        # 输出：状态字典新增chunks键；无local_dir则跳过备份，不影响主流程
        state["chunks"] = sections
        step_4_backup(state, sections)

        # 节点执行完成日志
        logger.info(f">>> 核心节点执行完成：【文档切分】{node_name}，已生成{len(sections)}个有效Chunk，结果已写入状态字典")

    except Exception as e:
        # 全局异常捕获：保证节点执行失败不崩溃整个流程，记录详细错误日志便于排查
        logger.error(f">>> 核心节点执行失败：【文档切分】{node_name}，错误信息：{str(e)}", exc_info=True)

    # 返回更新后的状态字典，传递Chunk结果到下游节点
    return state

if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册_new.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)
    with open(test_md_path,"r",encoding="utf-8") as f:
        md_content=f.read()
    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content":md_content ,
            "file_title": "hak180产品安全手册",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程



        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(test_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")