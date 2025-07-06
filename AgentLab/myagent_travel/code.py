import os
from typing import List, Dict,  TypedDict, Union
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from tavily import TavilyClient
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import re
from enum import Enum
from ratelimit import limits, RateLimitException
from backoff import on_exception, expo

os.environ["DASHSCOPE_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""

MEMORY_FILE = "long_term_memory.json"

llm = ChatOpenAI(
    model="gemini-2.5-flash",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    temperature=0.7,
)

# 配置检索API
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# 设置速率限制: 每60秒最多140次调用
@tool
@on_exception(expo, RateLimitException, max_tries=5)  
@limits(calls=140, period=60)
def tavily_search(query: str) -> str:
    """
    使用Tavily搜索引擎在线查找信息。
    """
    print(f"--- TOOL: TAVILY SEARCH --- \nQuery: {query}")
    try:
        results = tavily_client.search(query=query, search_depth="advanced", max_results=5)
        # 格式化搜索结果以便LLM理解
        formatted_results = "\n\n".join(
            [f"URL: {res['url']}\nTitle: {res['title']}\nContent: {res['content']}" for res in results['results']]
        )
        print(f"--- TOOL: TAVILY SEARCH --- \nResults: {formatted_results[:500]}...") 
        return formatted_results
    except RateLimitException:
        print("--- TOOL: TAVILY SEARCH --- \nRate limit reached, backing off...")
        raise  
    except Exception as e:
        return f"搜索失败，错误信息: {e}"

@tool
def search_long_term_memory(destination: str) -> Union[Dict, str]:
    """
    从长期记忆中搜索关于特定目的地的旅行计划。
    """
    print(f"--- TOOL: SEARCHING LTM for {destination} ---")
    try:
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            memory = json.load(f)
        if destination in memory:
            print(f"--- TOOL: Found plan for {destination} in LTM. ---")
            return memory[destination]
        else:
            return f"在长期记忆中没有找到关于 {destination} 的计划。"
    except FileNotFoundError:
        return "长期记忆文件不存在。"
    except Exception as e:
        return f"读取长期记忆时出错: {e}"

@tool
def add_to_long_term_memory(destination: str, plan_data: Dict) -> str:
    """
    将一个新的旅行计划添加到长期记忆中。
    """
    print(f"--- TOOL: ADDING to LTM: {destination} ---")
    try:
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                memory = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            memory = {}
            
        memory[destination] = plan_data
        
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(memory, f, ensure_ascii=False, indent=4)
        
        return f"成功将 {destination} 的计划添加到长期记忆中。"
    except Exception as e:
        return f"写入长期记忆时出错: {e}"


@tool
def calculator(expression: str) -> str:
    """
    一个简单的计算器工具，可以执行基本的数学运算。
    """
    print(f"--- TOOL: CALCULATOR --- \nExpression: {expression}")
    try:
        allowed_chars = "0123456789+-*/(). "
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"计算结果: {result}"
        else:
            return "无效的表达式，只允许数字和基本的运算符 (+, -, *, /, (, ))。"
    except Exception as e:
        return f"计算失败，错误信息: {e}"

@tool
def generate_markdown_report(plan: Dict) -> str:
    """
    使用LLM将结构化的旅行计划字典转换成Markdown格式报告（中文），并将其保存为文件。
    """
    print("--- TOOL: GENERATE MARKDOWN REPORT ---")
    
    prompt = f"""
    作为一名专业的旅行计划书撰写专家，请将以下旅行计划数据转换为美观、易读的中文Markdown格式报告。
    
    ## 旅行基本信息
    - 目的地: {plan['destination']}
    - 旅行时长: {plan['duration']}
    - 预算: {plan['budget']}
    - 旅客兴趣点: {', '.join(plan['interests'])}
    
    ## 行程安排
    {json.dumps(plan.get('itinerary', {}), ensure_ascii=False, indent=2)}
    
    ## 后勤信息
    {json.dumps(plan.get('logistics', {}), ensure_ascii=False, indent=2)}
    
    ## 备选方案
    {json.dumps(plan.get('alternatives', {}), ensure_ascii=False, indent=2)}
    
    请生成一份完整的、结构清晰的Markdown格式旅行计划，确保:
    1. 包含精美的标题和小标题（可以添加相关的图标或样式）
    2. 为每日行程添加详细描述，包括主题、活动和餐饮建议（详细）
    3. 格式化所有费用信息，使其易于阅读
    4. 将餐饮建议格式化为易读的列表，包含菜品推荐和评价（详细）
    5. 后勤部分应清晰展示住宿和交通信息
    6. 使用Markdown语法确保内容美观、层次分明
    
    只需返回纯Markdown格式，不要添加任何解释或前导语。
    """
    
    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    markdown_report = response.content

    try:
        filename = f"旅行计划_{plan.get('destination', '未知目的地')}.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        print(f"--- TOOL: REPORT EXPORTED --- \nSaved to: {filename}")
    except Exception as e:
        print(f"--- TOOL: REPORT EXPORT FAILED --- \nError: {e}")
    
    return markdown_report


class AgentType(str, Enum):
    CHIEF_PLANNER = "ChiefPlanner"
    ITINERARY_EXPERT = "ItineraryExpert"
    LOGISTICS_EXPERT = "LogisticsExpert"
    REPORT_GENERATOR = "ReportGenerator"
    END = "END"

class PlanDecision(BaseModel):
    """用于解析初始用户请求的结构。"""
    destination: str = Field(description="旅行目的地")
    duration: str = Field(description="旅行时间")
    budget: str = Field(description="旅行预算")
    interests: List[str] = Field(description="客户的兴趣点")
    next_agent: AgentType = Field(description="下一个应该开始工作的专家", default=AgentType.ITINERARY_EXPERT)

class ReviewDecision(BaseModel):
    """用于解析总监审查决策的结构。"""
    decision: str = Field(description="审查决定，'APPROVE' 或 'REVISE'")
    feedback: Union[str, None] = Field(description="如果决定是 'REVISE'，请提供具体的修改意见", default=None)
    next_agent: AgentType = Field(description="下一个智能体：'ReportGenerator' (如果批准) 或需要修改的专家 (如果修改)")


class AgentState(TypedDict):
    user_request: str                 # 用户的原始请求
    user_profile: Dict                # 提取出的用户核心偏好 (模拟长期记忆)
    plan: Dict                        # 存储旅行计划的结构化数据
    drafts: List[Dict]                # 各个智能体的草稿和思考过程
    reflection: Dict                  # 总监的反思和修改意见
    final_report: str                 # 最终的Markdown报告
    next_agent: str                   # 决定下一个执行的智能体


def call_model(state: AgentState, prompt: str, agent_name: str) -> Dict:
    """调用大模型并返回结构化的思考和响应"""
    print(f"\n--- AGENT: {agent_name} ---")
    messages = [HumanMessage(content=prompt)]
    # 将完整的思考过程和响应加入 'drafts'
    response = llm.invoke(messages)
    
    draft = {"agent": agent_name, "prompt": prompt, "response": response.content}
    state["drafts"].append(draft)
    print(f"Response:\n{response.content}")
    return state

def chief_planner_agent(state: AgentState) -> AgentState:
    """
    总监智能体 (ChiefPlanner)
    - 接收用户请求，提取核心信息
    - 审查草稿，提供反馈 (实现Reflection)
    - 决定下一步行动
    """
    if not state.get("plan"): # 任务开始
        parser = PydanticOutputParser(pydantic_object=PlanDecision)
        prompt = f"""
        你是一个高端旅行工作室的总监。你的任务是分析客户的初始请求，并委派任务给你的团队。
        
        客户请求: "{state['user_request']}"
        
        你的工作:
        1.  从请求中提取关键信息：目的地、时间、预算、个人兴趣等等。
        2.  决定下一步应该由哪个专家开始工作。
        
        {parser.get_format_instructions()}
        """
        # 创建链并调用
        chain = llm | parser
        parsed_output = chain.invoke(prompt)

        # 更新状态
        state["user_profile"] = {
            "destination": parsed_output.destination,
            "duration": parsed_output.duration,
            "budget": parsed_output.budget,
            "interests": parsed_output.interests
        }
        state["plan"] = {}
        
        print(f"--- CHIEF PLANNER: Checking long-term memory for {parsed_output.destination} ---")
        memory_result = search_long_term_memory.invoke({"destination": parsed_output.destination})
        
        if isinstance(memory_result, dict):
            print("--- CHIEF PLANNER: Found existing plan in memory. Using it as a base. ---")
            state['plan'] = memory_result.get('plan', {})
            state["next_agent"] = AgentType.CHIEF_PLANNER.value
        else:
            print("--- CHIEF PLANNER: No existing plan found. Starting from scratch. ---")
            state["next_agent"] = parsed_output.next_agent.value
        
    else: # Reflection
        itinerary_done = 'itinerary' in state['plan']
        logistics_done = 'logistics' in state['plan']
        
        if not itinerary_done:
            state["next_agent"] = AgentType.ITINERARY_EXPERT.value
            return state
        if not logistics_done:
            state["next_agent"] = AgentType.LOGISTICS_EXPERT.value
            return state

        parser = PydanticOutputParser(pydantic_object=ReviewDecision)
        prompt = f"""
        你是一个高端旅行工作室的总监，正在审查下属提交的旅行计划草稿。
        
        客户核心档案: {state['user_profile']}
        
        当前计划草稿: {state['plan']}
        
        你的任务 (Reflection):
        1. 检查 'itinerary' 是否详细且符合客户的兴趣({state['user_profile']['interests']})。
        2. 检查每个景点是否提供了地址，餐饮是否包含网友评论。
        3. 审查费用估算是否合理，每日费用总和是否在客户预算({state['user_profile']['budget']})范围内。
        4. 检查 'logistics' 的建议是否符合客户的预算。
        5. 如果一切都好，请决定下一步进入报告生成环节。输出 "APPROVE"。
        6. 如果有问题，请提出具体的修改意见和需要返工的专家名称。

        {parser.get_format_instructions()}
        """
        chain = llm | parser
        parsed_output = chain.invoke(prompt)

        if parsed_output.decision.upper() == "REVISE":
            state["next_agent"] = parsed_output.next_agent.value
            state['reflection'] = {"feedback": parsed_output.feedback}
        else:
            print("--- CHIEF PLANNER: Plan approved. Saving to long-term memory before generating report. ---")
            add_to_long_term_memory.invoke({
                "destination": state["user_profile"]["destination"],
                "plan_data": {
                    "user_profile": state["user_profile"],
                    "plan": state["plan"]
                }
            })
            state["next_agent"] = AgentType.REPORT_GENERATOR.value

    return state

def expert_agent_react(state: AgentState, agent_name: str) -> AgentState:
    """
    专家智能体 (行程/后勤)
    - 使用 ReAct 模式: 思考 -> 行动(工具) -> 观察 -> 回答
    - 它只负责自己领域内的规划。
    """
    task_description = ""
    if agent_name == "ItineraryExpert":
        task_description = f"""
        为期 {state['user_profile']['duration']} 的 {state['user_profile']['destination']} 之旅，规划一份详细的每日行程。
        客户对 {state['user_profile']['interests']} 特别感兴趣。
        总预算为 {state['user_profile']['budget']}。
        
        你需要：
        1. 为每个景点或地点查找并提供详细地址
        2. 为推荐的餐厅或小吃查找并提供真实的网友评论
        3. 估算每个活动和餐饮的费用，并计算每日总花费，确保不超过总预算
        """
        # 如果有总监的修改意见，加入到提示中
        if state.get('reflection') and 'ItineraryExpert' in state['reflection'].get('feedback', ''):
            task_description += f"\n修改意见: {state['reflection']['feedback']}"
            state['reflection'] = {} # 清除意见
            
    elif agent_name == "LogisticsExpert":
        task_description = f"""
        为 {state['user_profile']['destination']} 的旅行提供住宿和交通建议，总预算为 {state['user_profile']['budget']}。
        请提供住宿的具体地址、价格，以及从一个地点到另一个地点的交通方式和费用估算。
        """

    # ReAct Prompt
    prompt = f"""
    你是 {agent_name}。你的任务是: {task_description}
    
    请遵循"思考 -> 行动 -> 观察 -> 回答"的模式 (ReAct)。
    首先，你需要思考你需要哪些信息，然后使用你可用的工具({tavily_search.name}, {calculator.name})来查找这些信息或进行计算。
    在你收集到足够的信息后，请以JSON格式总结你的最终规划方案。JSON必须包含在一个代码块中。
    
    例如, 对于 ItineraryExpert, 最终的JSON输出格式应为:
    ```json
    {{
      "itinerary": {{
        "第一天": {{ "theme": "...", "activities": ["...", "..."], "dining_suggestion": "..." }},
        "第二天": {{ "theme": "...", "activities": ["...", "..."], "dining_suggestion": "..." }}
      }},
      "alternatives": {{ "备选活动": "..." }}
    }}
    ```
    
    对于 LogisticsExpert, 最终的JSON输出格式应为:
    ```json
    {{
      "logistics": {{
        "accommodation": "...",
        "transportation": "..."
      }},
      "alternatives": {{ "备选酒店": "..." }}
    }}
    ```

    现在，请开始你的工作。先思考，然后决定是否需要调用工具。
    """
    
    # ReAct 循环
    llm_with_tools = llm.bind_tools([tavily_search, calculator])
    messages = [HumanMessage(content=prompt)]
    for i in range(5): # 最多进行5轮
        print(f"\n--- {agent_name} ReAct Loop: Step {i+1} ---")
        
        response = llm_with_tools.invoke(messages)
        messages.append(response) # 将模型的响应加入历史记录

        # 检查模型是否要调用工具 (行动)
        if response.tool_calls:
            print(f"--- ReAct: Tool call detected by the model. ---")
            for tool_call in response.tool_calls:
                # 执行工具
                if tool_call['name'] == tavily_search.name:
                    tool_result = tavily_search.invoke(tool_call["args"])
                elif tool_call['name'] == calculator.name:
                    tool_result = calculator.invoke(tool_call["args"])
                else:
                    tool_result = "未知工具"
                    
                print(f"--- ReAct: Tool '{tool_call['name']}' executed. ---")
                # 将观察结果加入历史记录
                messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))
            continue

        # 如果没有工具调用，则认为是最终回答
        print("--- ReAct: No tool call detected. Assuming final answer. ---")
        json_match = re.search(r"```json\n(.*?)\n```", response.content, re.DOTALL)
        if json_match:
            plan_json_str = json_match.group(1)
            try:
                plan_update = json.loads(plan_json_str)
                state["plan"].update(plan_update)
                state["next_agent"] = AgentType.CHIEF_PLANNER.value
                print("--- ReAct: Successfully parsed final JSON answer. ---")
                return state
            except json.JSONDecodeError as e:
                print(f"--- ReAct: Error decoding JSON: {e} ---")
                # 告知模型JSON错误，让其修正
                error_message = HumanMessage(content=f"你的JSON格式有误，请修正: {e}")
                messages.append(error_message)
                continue

        # 如果既没有工具调用也没有有效JSON，让它再试一次
        print("--- ReAct: No valid action or final answer detected, retrying... ---")

    # 如果循环结束仍未得到有效结果
    print(f"--- {agent_name}: Failed to get a valid plan after multiple attempts. ---")
    state["next_agent"] = AgentType.CHIEF_PLANNER.value  # 返回给总监处理
    return state


def report_generator_agent(state: AgentState) -> AgentState:
    """
    报告生成专员
    - 调用自定义工具，将最终确认的计划格式化为Markdown
    """
    print("\n--- AGENT: ReportGenerator ---")
    
    complete_plan = {
        "destination": state["user_profile"]["destination"],
        "duration": state["user_profile"]["duration"],
        "budget": state["user_profile"]["budget"],
        "interests": state["user_profile"]["interests"]
    }

    complete_plan.update(state['plan'])
    

    markdown_report = generate_markdown_report.invoke({"plan": complete_plan})
    state['final_report'] = markdown_report
    state['next_agent'] = AgentType.END.value  # 任务结束
    return state



def route_tasks(state: AgentState) -> str:
    """根据 'next_agent' 字段的值决定下一个节点"""
    next_agent = state.get("next_agent", AgentType.CHIEF_PLANNER.value)
    print(f"--- ROUTER: Decided next agent is {next_agent} ---")
    return next_agent

workflow = StateGraph(AgentState)

workflow.add_node("ChiefPlanner", chief_planner_agent)
workflow.add_node("ItineraryExpert", lambda state: expert_agent_react(state, "ItineraryExpert"))
workflow.add_node("LogisticsExpert", lambda state: expert_agent_react(state, "LogisticsExpert"))
workflow.add_node("ReportGenerator", report_generator_agent)

workflow.set_entry_point("ChiefPlanner")

workflow.add_conditional_edges(
    "ChiefPlanner",
    route_tasks,
    {
        "ItineraryExpert": "ItineraryExpert",
        "LogisticsExpert": "LogisticsExpert",
        "ReportGenerator": "ReportGenerator",
    }
)
workflow.add_edge("ItineraryExpert", "ChiefPlanner")
workflow.add_edge("LogisticsExpert", "ChiefPlanner")

workflow.add_edge("ReportGenerator", END)

graph = workflow.compile()



def main():
    print("\n=== 高端旅行定制工作室开始工作 ===")
    
    initial_request = "你好，我和我男朋友想去中国广东潮汕玩四天。预算在三千左右。我们对当地美食非常感兴趣，同时也想参观当地一些名胜景地，希望能有一个深度游的体验。"
    
    initial_state = AgentState(
        user_request=initial_request,
        user_profile={},
        plan={},
        drafts=[],
        reflection={},
        final_report="",
        next_agent=AgentType.CHIEF_PLANNER.value
    )
    
    print(f"\n客户初始请求: {initial_request}\n")
    print("="*20)
    
    final_state = None
    for output in graph.stream(initial_state, config={"recursion_limit": 100}):
        if "__end__" not in output:
            for key, value in output.items():
                print(f"--- Node '{key}' Output ---")
                final_state = value
    
    print("\n\n" + "="*50)
    print("      旅行计划制定完成!      ")
    print("="*50)

    # 打印最终的Markdown报告
    print(final_state['final_report'])


if __name__ == "__main__":
    main()