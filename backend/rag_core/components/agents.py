"""Prompt chains for routing, planning, research and the shared final writer."""

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from ..models import RouteQuery, Plan, BilingualQuery, PolicyDecision

def build_router_agent(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一名专家级的意图分类器。你的任务是根据问题的复杂度，选择最高效的解决路径。

        路径选择标准:

        1.  SIMPLE_ANSWER (快速检索模式):
            - 定义: 问题具体、明确，只需要进行一次数据库检索即可找到答案，不需要多步推理或复杂的横向对比。
            - 适用场景:
                - 询问具体事实 (如 "Meta Free Transformer的架构细节", "夸克眼镜哪天预售", "Kyber项目何时量产")。
                - 简单的概念解释 (如 "什么是RAG", "介绍一下Transformer")。
            - 决策: 将 `step` 设为 "simple_answer"。即使需要查库，只要一次能查到，就选这个。

        2.  COMPLEX_RESEARCH (深度思考模式):
            - 定义: 问题模糊、庞大，或者需要跨多个时间段、多个实体进行综合分析，往往需要拆解为多个子问题。
            - 适用场景:
                - 趋势分析 (如 "分析2025年AI发展趋势")。
                - 深度对比 (如 "详细对比DeepSeek和ChatGPT在编码能力上的优劣及底层架构差异")。
                - 综述报告 (如 "总结过去一个月所有关于半导体的大事件")。
            - 决策: 将 `step` 设为 "complex_research"。

        输出格式:
        严格遵循JSON格式。
        {{
            "step": "simple_answer" 或 "complex_research",
            "simple_response": ""  <-- 永远留空，因为我们现在改用检索生成，不再由你直接回答
        }}
        {format_instructions}"""),
        ("human", "用户问题: {question}")
    ])
    parser = JsonOutputParser(pydantic_object=RouteQuery)
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser

def build_planner_agent(llm):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "你是一名顶级的AI查询意图分析师。你的任务是将用户的复杂问题分解为一个结构清晰、逻辑严谨的多步骤研究计划。"
         "专注于问题分解，将数据源选择等策略性任务留给下游Agent。你的输出必须是严格的JSON格式，完全符合以下Pydantic模型。\n{format_instructions}"),
        ("human", "用户问题: {question}")
    ])
    parser = JsonOutputParser(pydantic_object=Plan)
    return prompt_template.partial(format_instructions=parser.get_format_instructions()) | llm | parser

def build_universal_query_rewriter_agent(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个为多语言数据库设计的、专家级的查询优化引擎。

        **核心任务**:
        1.  **查询优化**: 接收用户查询，生成两个优化版本（中文和英文），使其更适合数据库检索。
        2.  **实体提取**: 识别并提取查询中的关键技术实体（如公司名、AI模型、硬件型号、算法名称）和专有名词。
        3.  **时间识别**: 如果查询包含明确的时间范围（如"上周"、"2025年"），将其提取出来。

        **输出格式要求**:
        你的回答必须严格遵循以下JSON格式，不允许包含额外文本。
        ```json
        {{
            "chinese_query": "优化后的中文查询",
            "english_query": "Optimized English query",
            "tech_entities": ["实体1", "实体2"],
            "time_range": "提取的时间描述 或 null"
        }}
        ```

        **示例 1 (中文输入)**:
        - 输入: 子问题="对比DeepSeek和GPT-4在编码上的表现", 关键词="DeepSeek, GPT-4, 编码"
        - 输出:
        ```json
        {{
            "chinese_query": "对比并评测 DeepSeek 和 GPT-4 在代码编写和编程能力上的表现差异",
            "english_query": "DeepSeek vs GPT-4 coding performance benchmark",
            "tech_entities": ["DeepSeek", "GPT-4"],
            "time_range": null
        }}
        ```

        **示例 2 (英文输入)**:
        - 输入: 子问题="What happened to NVIDIA stock last week?", 关键词="NVIDIA, stock"
        - 输出:
        ```json
        {{
            "chinese_query": "英伟达 股价 波动 上周 新闻",
            "english_query": "NVIDIA stock price fluctuations last week news",
            "tech_entities": ["NVIDIA"],
            "time_range": "last week"
        }}
        ```

        请严格遵守以上规则。"""),
        ("human", "用户查询: {user_query}\n\n相关关键词: {keywords}\n\n历史上下文:\n{past_context}")
    ])

    parser = JsonOutputParser(pydantic_object=BilingualQuery)
    return prompt | llm | parser

def build_distiller_agent(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一位乐于助手的AI。你的任务是将以下检索到的文档片段合成为一个单一、简洁的段落。目标是提供一个清晰、连贯的上下文，直接回答问题：'{question}'。专注于消除冗余信息并按逻辑组织内容。只回答合成后的上下文。"),
        ("human", "检索到的文档:\n{context}")
    ])
    return prompt | llm | StrOutputParser()

def build_reflection_agent(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一名研究助理。根据当前子问题和提炼后的上下文，写一个简洁、事实驱动的单句摘要，总结核心发现。"),
        ("human", "当前子问题: {sub_question}\n\n提炼后的上下文:\n{distilled_context}")
    ])
    return prompt | llm | StrOutputParser()

def build_policy_agent(llm):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "你是一位战略大师，负责分析研究进展并决定下一步行动。分析依据: 1. 原始问题, 2. 初始计划, 3. 研究历史。"
         "决策: 如果历史信息已足够全面回答原始问题，决策为 FINISH。否则，决策为 CONTINUE。你的输出必须是严格的JSON格式。\n{format_instructions}"),
        ("human", "原始问题: {question}\n\n初始计划:\n{plan}\n\n研究历史:\n{history}")
    ])
    parser = JsonOutputParser(pydantic_object=PolicyDecision)
    return prompt_template.partial(format_instructions=parser.get_format_instructions()) | llm | parser

def get_final_answer_prompt():
    """暴露统一的最终生成提示词，供服务端流式输出与实验复用。"""
    return ChatPromptTemplate.from_messages([
        ("system", """你是一位顶尖的科技动向分析师。你的任务是基于下方提供的、已编号的上下文信息，为用户的原始问题生成一份精准、深入且结构清晰的最终报告。

        ！！！核心规则 - 必须严格遵守！！！:
        1.  绝对忠于原文: 你的回答必须完全基于提供的上下文。严禁使用外部知识。
        2.  句句皆有出处: 报告中的每一个句子或独立信息点，都必须在其末尾使用 `[来源:INDEX]` 的格式清晰标注其来源的文档索引。
        3.  多来源标注: 如果一个句子综合了多个来源的信息，请使用逗号分隔索引，例如 `[来源:2, 5]`。
        4. 格式要求: 
           - 使用Markdown标题、要点和列表来组织回答，使其清晰易读。
           - 严禁使用任何 HTML 标签（如 <span>, <div>, <script> 等）。
           - 引用标注必须严格使用纯文本格式：[来源:1] 或 [来源:1, 2]。
           - 绝对不要输出任何 HTML 或 JavaScript 代码。
        5.  无来源则不答: 如果提供的上下文信息不足以回答问题的某个方面，请明确指出“根据提供的资料，无法回答关于[某方面]的问题”，不要进行猜测。
        6.  格式示例:
            - Meta在2025年10月22日进行了大规模裁员 [来源:1, 3]。
            - 新成立的TBD Lab将专注于下一代AI模型的研发 [来源:4]。

        开始分析以下上下文，并生成报告。"""),
        ("human", "原始问题: {question}\n\n已编号的上下文信息:\n{context}")
    ])
