# tools.py
from rag import RAG

# 懒加载 RAG（第一次调用 rag_search 时再初始化）
rag = None
"""
工具模块：定义所有可供 Agent 调用的工具
每个工具包含：
- func: 实际执行函数
- schema: 给 LLM 的结构化描述（用于生成JSON）
"""

def calculator(expression: str) -> str:
    """
    计算数学表达式
    
    参数:
        expression: 字符串形式的数学表达式，例如 "12*45"
    
    返回:
        计算结果（字符串）
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {str(e)}"


def mock_search(query: str) -> str:
    """
    模拟搜索工具（实际项目可以接入搜索API）
    
    参数:
        query: 搜索关键词
    
    返回:
        模拟搜索结果
    """
    return f"搜索结果：关于『{query}』的相关信息..."

def rag_search(query: str) -> str:
    """
    从本地知识库检索信息
    """
    global rag
    if rag is None:
        print("RAG 初始化中...")
        rag = RAG()
        print("RAG 初始化完成")
    return rag.search(query)


# 工具注册表（核心）
tools = {
    "calculator": {
        "func": calculator,
        "schema": {
            "name": "calculator",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 12*45"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    "search": {
        "func": mock_search,
        "schema": {
            "name": "search",
            "description": "搜索信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索内容"
                    }
                },
                "required": ["query"]
            }
        }
    },
    "rag_search": {
        "func": rag_search,
        "schema": {
            "name": "rag_search",
            "description": "从本地知识库向量检索；返回条数由环境变量 RAG_TOP_K 控制（默认 8），切块大小由 RAG_CHUNK_SIZE 控制（改后需重建索引）",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "查询内容"
                    }
                },
                "required": ["query"]
            }
        }
    }
}