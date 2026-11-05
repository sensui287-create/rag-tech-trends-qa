"""Validated agent outputs and the state passed between graph nodes."""

from typing import List, Optional, TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_core.documents import Document

class RouteQuery(BaseModel):
    step: Literal["simple_answer", "complex_research"] = Field(
        description="下一步行动。'simple_answer' 用于单次检索可回答的问题，'complex_research' 用于需要多步骤研究的问题。"
    )
    simple_response: str = Field(
        default="",
        description="保留字段，始终留空；最终回答由检索与生成流程完成。"
    )

class BilingualQuery(BaseModel):
    chinese_query: str = Field(description="优化后的中文搜索查询")
    english_query: str = Field(description="翻译并优化后的英文搜索查询")
    tech_entities: List[str] = Field(default_factory=list, description="查询中的关键技术实体")
    time_range: Optional[str] = Field(default=None, description="查询中的时间范围")

class Step(BaseModel):
    sub_question: str = Field(description="具体子问题")
    justification: str = Field(description="必要性解释")
    keywords: List[str] = Field(description="关键术语列表")

class Plan(BaseModel):
    steps: List[Step] = Field(min_length=1, description="多步骤研究计划（至少包含一个步骤）")

class PolicyDecision(BaseModel):
    next_action: str = Field(description="'CONTINUE' 或 'FINISH'")
    justification: str = Field(description="理由")

class PastStep(TypedDict):
    step_index: int
    sub_question: str
    sub_question_rewritten: str
    retrieved_docs: List[Document]
    distilled_context: str
    summary: str

class AgentState(TypedDict):
    original_question: str
    router_decision: str
    plan: Plan
    past_steps: List[PastStep]
    current_step_index: int
    current_rewritten_query: str
    retrieved_docs: List[Document]
    distilled_context: str
    candidate_docs: List[Document]
    final_answer: str
    context_for_final_answer: Optional[str]
    metadata_for_final_answer: Optional[dict]
