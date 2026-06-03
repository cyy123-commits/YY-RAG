# 系统库
import os
import sys
import time
import requests
import zipfile
import shutil
from pathlib import Path



# 项目内部库
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.format_utils import format_state
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger  # 统一日志工具
from app.conf.mineru_config import mineru_config
# MinerU配置（缓存配置信息）
MINERU_BASE_URL = mineru_config.base_url
MINERU_API_TOKEN = mineru_config.api_key


def step_1_validate_paths(state):
    """
    步骤1：校验PDF文件路径和输出目录
    核心职责：参数非空校验 | PDF文件有效性校验 | 输出目录自动创建
    返回：合法的PDF文件Path对象、输出目录Path对象
    异常：ValueError(参数缺失)、FileNotFoundError(文件无效)
    """
    log_prefix = "[step_1_validate_paths] "
    pdf_path = state["pdf_path"]
    local_dir = state["local_dir"]

    # 参数非空校验
    if not pdf_path:
        raise ValueError(f"{log_prefix}检查发现没有输入文件，无法继续解析")
    if not local_dir:
        local_dir=PROJECT_ROOT/"output"
        raise ValueError(f"{log_prefix}检查发现local_dir没有赋值，给与默认值：{local_dir}")

    # 转换为Path对象统一处理路径
    pdf_path_obj = Path(pdf_path)
    output_dir_obj = Path(local_dir)

    # PDF文件有效性校验（存在且为文件，非目录）
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"{log_prefix}PDF文件不存在，绝对路径：{pdf_path_obj.absolute()}")
    if not pdf_path_obj.is_file():
        raise FileNotFoundError(f"{log_prefix}指定路径非文件（是目录），绝对路径：{pdf_path_obj.absolute()}")

    # 确保输出目录存在，不存在则递归创建
    if not output_dir_obj.exists():
        logger.info(f"{log_prefix}输出目录不存在，自动创建：{output_dir_obj.absolute()}")
        output_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj, output_dir_obj


def step_2_upload_and_poll(pdf_path_obj: Path):
    """

    :param pdf_path_obj: 上传解析的pdf文件的path对象
    :return:minerU解析后md文件zip压缩包的下载地址
    """
    #动态获取函数名，避免硬编码
    func_name = sys._getframe().f_code.co_name
 #1。申请上传解析的地址
    #前置准备和参数 url api | token |准备固定格式的请求头

    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"

    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": f"{pdf_path_obj.name}"}
        ],
        "model_version": "vlm"
    }
    response = requests.post(url, json=data, headers=header)
    try:
        json_data = response.json()
    except Exception:
        json_data = {}

    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP请求失败，status={response.status_code}, "
            f"response={response.text}"
        )

    if json_data.get("code") != 0:
        raise RuntimeError(
            f"MinerU业务失败，response={json_data}"
        )
    upload_url = response.json()["data"]["file_urls"][0]#去这个地址上传文件
    batch_id=response.json()["data"]["batch_id"]#处理id，后续根据这个id获取结果

#2.将文件上传到对应解析地址。。
    #使用Put请求，讲pdf_path_obj文件上传到upload_url上，需要注意的是不能直接使用put，用代理的话大概率报错，put严进严出
    http_session=requests.Session()#创建http客户端，后面需要频繁查询结果，就不用经常发起请求了，这里主要用来禁止使用代理
    http_session.trust_env=False#禁止使用代理，还有一个功能是复用请求对象，这里主要用来禁止使用代理
    try:
        with open(pdf_path_obj, "rb") as f:
            file_data=f.read()
        upload_response=http_session.put(upload_url,data=file_data)
        if upload_response.status_code != 200:
            logger.error(f"{func_name}上传文件到minerU失败，请检查输入路径是否正确")
            raise RuntimeError(f"{func_name}上传文件到minerU失败，请检查输入路径是否正确")
    except Exception as e:
        logger.error(f"{func_name}上传文件到minerU失败，请检查输入路径是否正确")
        raise RuntimeError(f"{func_name}上传文件到minerU失败，请检查输入路径是否正确")

    finally:
        http_session.close()
#3.轮询获取解析结果
    #循环获取，确保获取到结果，再先后执行
    #设计一个循环，sleep3秒获取一次，最多等待10分钟，600s，一般一秒一页pdf
    poll_interval=3
    start_time = time.time()

    url2 = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    while True:

        if time.time() - start_time > 600:
            logger.error(f"{func_name}请求minerU接口超时")
            raise TimeoutError(f"{func_name}请求minerU接口超时")
        res=requests.get(url2,headers=header)

        if res.status_code !=200:
            #5开头，给机会继续等，知道超时
            if 500<=res.status_code<600:
                time.sleep(poll_interval)
                continue
            raise RuntimeError(f"{func_name}请求minerU解析接口失败，返回的状态是{res.status_code}")
        json_data=res.json()
        #判断结果code
        if json_data['code']!=0:
            #大概token过期或者后续没钱了
            raise RuntimeError(f"{func_name}请求minerU接口失败，返回{json_data['code']}")

        #判断解析状态
        extract_result=json_data['data']['extract_result'][0]
        if extract_result["state"]=="done":
            #解析完了，可以获取结果了
            full_zip_url=extract_result["full_zip_url"]
            logger.info(f"已完成pdf解析，耗时：{time.time()-start_time}s,解析结果链接为{full_zip_url}")
            return full_zip_url

        else:
            time.sleep(poll_interval)





def step_3_download_and_extract(zip_url: str, local_dir_obj: Path, pdf_stem: str) -> str:
    """
    步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件（重命名统一规范）
    核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件（按优先级） → 重命名统一为PDF同名
    参数：zip_url-ZIP包下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
    返回：最终MD文件的字符串格式绝对路径
    异常：RuntimeError(下载失败)、FileNotFoundError(无MD文件)
    """
    func_name=func_name = sys._getframe().f_code.co_name

    #1。下载zip文件 reponse响应体
    response=requests.get(zip_url)

    if response.status_code!=200:
        logger.error(f"{func_name},下载zip文件失败")
        raise RuntimeWarning(f"{func_name},下载zip文件失败")
    #2.将响应体的zip文件保存到本地
    zip_save_path=local_dir_obj/f"{pdf_stem}_result.zip"
    with open(zip_save_path,"wb")as f:
        f.write(response.content)#content会返回响应体中二进制字节，文件，图片等
    logger.info(f"{func_name}下载文件成功，保存位置L{zip_save_path}")
    #3清空旧目录（将上一次处理的文件目录删除）
    extract_target_dir=local_dir_obj/pdf_stem  #我这里是再output目录下先放一个zip包，解压以后就会在output下出来一个同名文件夹，所以是清空这个文件夹
    if extract_target_dir.exists():
        shutil.rmtree(extract_target_dir)

    #创建一个新目录
    extract_target_dir.mkdir(parents=True,exist_ok=True)
    #4.进行zip文件解压    这里用zipfile这个python解压的模块，创建一个zipfile对象，只读的方式解压
    with zipfile.ZipFile(zip_save_path,"r") as zip_file_object:
        zip_file_object.extractall(extract_target_dir)#解压到的目标目录

    #4.返回md文件的地址，
    # 解压后的文件中的md文件可能叫stem.md或者full.md等其他乱七八糟的，需要统一
    md_file_list=list(extract_target_dir.rglob("*.md"))
    if not md_file_list:
        logger.error(f"{func_name}没有找到md文件，请检查输入文件路径是否正确")
        raise RuntimeError(f"{func_name}没有找到md文件，请检查输入文件路径是否正确")

    target_md_file=None
    for md_file in md_file_list:
        if md_file.name == pdf_stem :
            target_md_file=md_file
            break
    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name.islower()=="full.md":
                target_md_file=md_file
                break

    if not target_md_file:
        target_md_file=md_file_list[0]

    #统一讲target_md_file改成{pdf.stem}.md
    #从内向外先with_mame重新命名path对象名字，然后用这个名字rename md文件路径名字
    if target_md_file.stem != pdf_stem:
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")

        target_md_file.rename(new_md_path)

        final_md_file_path_str = str(new_md_path)
    #下面要把最终的md文件路径赋值给state中，所以要变成字符串了
    logger.info(f"{func_name}完成了zip解压，最终存储路径是{final_md_file_path_str}")

    return final_md_file_path_str




def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    LangGraph工作流节点：PDF转MD核心处理节点
    核心流程：路径校验 → MinerU上传解析 → 结果下载解压 → 读取MD内容并更新工作流状态
    参数：state-工作流状态对象，需包含pdf_path/local_dir/task_id
    返回：更新后的工作流状态，新增md_path/md_content
    """

    # 动态获取函数名避免硬编码
    func_name = sys._getframe().f_code.co_name

    # 节点启动日志，打印当前工作流状态
    logger.debug(f"【{func_name}】节点启动，\n当前工作流状态：{format_state(state)}")

    # 开始：记录节点运行状态
    add_running_task(state["task_id"], func_name)


    try:
        # 步骤1：校验PDF路径和输出目录
        pdf_path_obj, output_dir_obj = step_1_validate_paths(state)

        # 步骤2：上传PDF至MinerU并轮询解析结果
        zip_url = step_2_upload_and_poll(pdf_path_obj)

        # 步骤3：下载ZIP包并提取MD文件
        md_path = step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)

        # 更新工作流状态：记录MD文件路径和内容
        state["md_path"] = md_path
        logger.info(f"【{func_name}】MD文件生成成功，路径：{md_path}")

        # 读取MD文件内容，捕获异常仅警告不终止
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                state["md_content"] = f.read()
            logger.debug(f"【{func_name}】MD文件内容读取成功，内容长度：{len(state['md_content'])}字符")
        except Exception as e:
            logger.error(f"【{func_name}】读取MD文件内容失败：{str(e)}")

        logger.info(f"【{func_name}】节点执行完成，更新后状态为：{format_state(state)}")

    except Exception as e:
        # 异常日志分级，精准提示配置问题
        logger.error(f"【{func_name}】PDF转MD流程执行失败：{str(e)}", exc_info=True)
        raise  # 抛出异常，终止工作流
    finally:

        # 结束：记录节点运行状态
        add_done_task(state["task_id"], func_name)




    return state

if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")