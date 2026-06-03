import sys
import json
import asyncio


from app.utils.task_utils import add_done_task, add_running_task
from app.conf.bailian_mcp_config import mcp_config
from agents.mcp import MCPServerStreamableHttp
from app.core.logger import logger
#mcp sdk调用url以及apikey
DASHSCOPE_BASE_URL_STREAMBLE=mcp_config.mcp_base_url
DASHSCOPE_API_KEY=mcp_config.api_key




async def mcp_call_streamble(query):
    """
    调用网络搜索工具

    :param query:
    :return:
    """

    #1.创建MCPServerStreambleHttp对象
    search_mcp=MCPServerStreamableHttp(
        name='search_mco',
        params={
            "url":DASHSCOPE_BASE_URL_STREAMBLE,
            "headers":{"Authorization":f"Bearer {DASHSCOPE_API_KEY}"},
            "timeout":10
        },
        max_retry_attempts=3
    )

    #2.连接    调用    关闭
    try:
        await search_mcp.connect()

        tools = await search_mcp.list_tools()
        print(f"可用工具: {[tool.name for tool in tools]}")

        #调用
        result=await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query":query,
                "count":5
            }
        )
        return result
    except Exception as e:
        logger.exception(f"连接或调用mcp客户端失败")


    finally:
        await search_mcp.cleanup()






def node_web_search_mcp(state):
    """
    节点功能：调用外部搜索补充信息
    :param state: 
    :return: 
    """""

    add_running_task(state["session_id"],sys._getframe().f_code.co_name,state.get("is_stream"))



    #1.获取参数信息
    query=state.get("rewritten_query")

    #2.调用streamable的网络搜索方法
    result=asyncio.run(mcp_call_streamble(query))

    """
    result = MCPCallToolResult(
    # 元数据信息（通常为 None）
    meta=None,
    
    # 内容列表，包含 TextContent 对象
    content=[
        TextContent(
            type='text',  # 内容类型
            text='{"pages": [...], "request_id": "...", "tools": [], "status": 0}',  # JSON 字符串
            annotations=None,  # 注解信息
            meta=None  # 元数据
        )
    ],
    
    # 结构化内容（通常为 None）
    structuredContent=None,
    
    # 是否为错误响应
    isError=False
)
    
    
    """

    #3.处理结果
    web_documents=json.loads(result.content[0].text).get("pages",[])
    #这里的结果是pages对应的列表：[{snippet:内容，title:标题，url:关联的文章或者图片的地址}，{},{},{},{}]

    logger.info(f"mcp搜索方法结果为：{web_documents}")
    add_done_task(state["session_id"],sys._getframe().f_code.co_name,state.get("is_stream"))
    return {"web_documents":web_documents}
if __name__ == '__main__':
    # 测试代码：单独运行该文件时，验证MCP搜索功能是否正常
    print("\n" + "="*50)
    print(">>> 启动 node_web_search_mcp 本地测试")
    print("="*50)

    test_state = {
        "session_id": "test_mcp_session",
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
        "is_stream": False
    }

    try:
        # 调用MCP搜索节点函数，执行测试
        result_state = node_web_search_mcp(test_state)

        print("\n" + "="*50)
        print(">>> 测试结果摘要:")
        search_results = result_state.get('web_documents', [])
        print(f"搜索结果数量: {len(search_results)}")
        if search_results:
            print("首条结果预览:")
            print(json.dumps(search_results[0], indent=2, ensure_ascii=False))
        else:
            print("未获取到搜索结果")
        print("="*50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
