import os
from typing import List, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

from langgraph.graph import StateGraph, END

os.environ["GOOGLE_API_KEY"] = ""
os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'
os.environ["TAVILY_API_KEY"] = ""

# --- 1. 定义图的状态 (State) ---
# State是整个流程中数据的载体，可以理解为每个Agent共享的“短期记忆”。
class GraphState(TypedDict):
    topic: str  # 用户输入的主题
    outline: str  # 由主编Agent生成的大纲
    research_data: List[dict]  # 由搜索Agent收集的资料
    draft: str  # 由撰稿人Agent生成的初稿
    reflection: str # 由事实核查Agent生成的反思/修改建议
    final_report: str # 最终生成的报告

# --- 2. 准备工具 (Tools) ---
# 这里我们使用Tavily作为搜索工具
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# --- 3. 初始化大语言模型 (LLM) ---
# 我们将使用Google的Gemini模型来驱动所有的Agent
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)


# --- 4. 定义Agent的输出结构 (Pydantic Models) ---
# 使用Pydantic可以强制LLM输出我们想要的JSON格式，方便程序解析。

class ReportOutline(BaseModel):
    """报告大纲的结构"""
    outline: str = Field(description="报告的章节大纲，使用Markdown格式。")

class ResearchInfo(BaseModel):
    """研究信息的结构"""
    results: List[dict] = Field(description="一个包含字典的列表，每个字典代表一条搜索结果，包含'url'和'content'。")

class ReportDraft(BaseModel):
    """报告初稿的结构"""
    draft: str = Field(description="报告的完整初稿，使用Markdown格式。")
    
class Reflection(BaseModel):
    """反思与修改建议的结构"""
    reflection_notes: str = Field(description="关于初稿的修改建议。如果没有问题，请返回'OK'。")
    is_ok: bool = Field(description="初稿是否通过核查，无需修改。")


# --- 5. 创建各个Agent的节点 (Nodes) ---

def chief_editor_node(state: GraphState):
    """
    主编Agent节点
    功能：根据用户主题，生成报告大纲。
    """
    print(">> 进入 [主编Agent] 节点")
    prompt = ChatPromptTemplate.from_template(
        """你是一位资深的新闻主编。你的任务是为一个给定的主题创建一个简洁、全面、逻辑清晰的报告大纲。
        主题: {topic}"""
    )
    # .with_structured_output() 会让LLM的输出自动格式化为我们定义的Pydantic模型
    chain = prompt | llm.with_structured_output(ReportOutline)
    result = chain.invoke({"topic": state["topic"]})
    
    print(f"   - 生成的大纲:\n{result.outline}")
    return {"outline": result.outline}

def search_agent_node(state: GraphState):
    """
    搜索Agent节点
    功能：根据主题和大纲，使用Tavily搜索相关信息。
    """
    print(">> 进入 [搜索Agent] 节点")
    # 为了简化，我们直接用主题进行搜索，更复杂的实现可以结合大纲生成更具体的搜索查询
    search_query = f"关于 {state['topic']} 的最新进展和详细信息"
    print(f"   - 正在执行搜索: {search_query}")
    
    # Tavily的 `search` 方法返回丰富的搜索结果
    response = tavily_client.search(query=search_query, search_depth="advanced", max_results=5)
    
    # 将结果格式化为我们需要的字典列表
    research_data = [{"url": res["url"], "content": res["content"]} for res in response["results"]]
    
    print(f"   - 找到 {len(research_data)} 条相关信息")
    return {"research_data": research_data}

def writer_agent_node(state: GraphState):
    """
    撰稿人Agent节点
    功能：根据大纲和研究资料撰写报告初稿。
    """
    print(">> 进入 [撰稿人Agent] 节点")
    
    # 如果有反思（修改意见），则需要根据修改意见来重新撰写
    if state.get("reflection"):
        print("   - 检测到修改意见，正在进行修订...")
        prompt_template = """你是一位专业的新闻撰稿人。请根据以下大纲、研究资料和修改建议，重新撰写一份详细、客观、高质量的新闻报告。
        
        【原始大纲】
        {outline}
        
        【研究资料】
        {research_data}
        
        【必须遵守的修改建议】
        {reflection}
        
        请输出一份完整的、修订后的报告。
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm.with_structured_output(ReportDraft)
        result = chain.invoke({
            "outline": state["outline"],
            "research_data": state["research_data"],
            "reflection": state["reflection"]
        })
    else:
        # 首次撰写
        print("   - 正在撰写初稿...")
        prompt_template = """你是一位专业的新闻撰稿人。请根据以下大纲和研究资料，撰写一份详细、客观、高质量的新闻报告。
        
        【报告大纲】
        {outline}
        
        【研究资料】
        {research_data}
        
        请确保报告内容完全基于提供的研究资料，并遵循大纲结构。
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | llm.with_structured_output(ReportDraft)
        result = chain.invoke({
            "outline": state["outline"],
            "research_data": str(state["research_data"]) # 转为字符串方便处理
        })

    print("   - 初稿/修订稿完成。")
    return {"draft": result.draft}

def fact_checker_node(state: GraphState):
    """
    事实核查Agent节点 (实现Reflection)
    功能：核查初稿内容是否与原始资料一致，并提供修改建议。
    """
    print(">> 进入 [事实核查Agent & 反思] 节点")
    prompt = ChatPromptTemplate.from_template(
        """你是一名严谨的事实核查员。你的任务是仔细阅读报告初稿，并将其与原始研究资料进行比对。
        检查报告中是否有任何不准确、夸大或缺乏依据的陈述。
        
        - 如果报告内容准确无误，忠于原文，请在修改建议中仅返回 "OK"。
        - 如果发现问题，请清晰地指出问题所在，并提出具体的修改建议。
        
        【报告初稿】
        {draft}
        
        【原始研究资料】
        {research_data}
        """
    )
    chain = prompt | llm.with_structured_output(Reflection)
    result = chain.invoke({
        "draft": state["draft"],
        "research_data": str(state["research_data"])
    })
    
    if result.is_ok:
        print("   - 核查结果: [通过]")
        # 如果通过，将草稿定为最终报告
        return {"final_report": state["draft"], "reflection": ""}
    else:
        print(f"   - 核查结果: [需要修改]")
        print(f"   - 修改建议: {result.reflection_notes}")
        # 如果不通过，将修改意见存入state，以便撰稿人进行修改
        return {"reflection": result.reflection_notes}


# --- 6. 定义图的逻辑流 (Edges) ---

def should_revise_edge(state: GraphState):
    """
    条件判断边
    功能：根据事实核查Agent的结果，决定是返回修改还是结束流程。
    """
    print(">> 进行条件判断: 是否需要修订？")
    if state.get("reflection"):
        print("   - 决策: 是，返回撰稿人进行修改。")
        return "revise" # 返回一个字符串，对应我们添加的条件边的名称
    else:
        print("   - 决策: 否，流程结束。")
        return END


# --- 7. 构建并编译图 (Graph) ---
# 这是一个有向无环图，但通过条件边可以实现循环
workflow = StateGraph(GraphState)

# 添加节点
workflow.add_node("chief_editor", chief_editor_node)
workflow.add_node("searcher", search_agent_node)
workflow.add_node("writer", writer_agent_node)
workflow.add_node("fact_checker", fact_checker_node)

# 设置入口点
workflow.set_entry_point("chief_editor")

# 添加边
workflow.add_edge("chief_editor", "searcher")
workflow.add_edge("searcher", "writer")
workflow.add_edge("writer", "fact_checker")

# 添加条件边（实现Reflection循环）
workflow.add_conditional_edges(
    "fact_checker",
    should_revise_edge,
    {
        "revise": "writer", # 如果`should_revise_edge`返回"revise"，则下一个节点是`writer`
        END: END            # 如果返回END，则流程结束
    }
)

# 编译图
app = workflow.compile()


# --- 8. 运行图并查看结果 ---

if __name__ == "__main__":
    # 定义一个输入，这个字典的key必须和GraphState中的key对应
    inputs = {"topic": "英伟达Blackwell架构的最新进展"}
    
    # LangGraph的 `stream` 方法会返回每一步的结果，方便我们观察流程
    for output in app.stream(inputs, stream_mode="values"):
        # `output` 是一个字典，key是节点名，value是该节点的返回结果
        # 我们可以打印出来看看每一步发生了什么
        print("---")
        # print(output) # 打印原始输出
    
    # `stream` 结束后，最后一个 `output` 值就是整个图的最终状态
    final_state = output
    
    print("\n\n✅ 报告生成完毕！")
    print("=" * 30)
    print(final_state["final_report"])

    # 简单演示长期记忆的理念
    # 在实际应用中，这里会把 final_state["final_report"] 存储到向量数据库中
    long_term_memory = []
    long_term_memory.append({
        "topic": final_state["topic"],
        "report": final_state["final_report"]
    })
    print("\n\n🧠 [演示] 报告已存入'长期记忆'。")