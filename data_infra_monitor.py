#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 기반 데이터 인프라 모니터링 서비스
"""

import os
import json
import re
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Optional, Literal, Annotated
from operator import add

# LangChain 관련 imports
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangGraph 관련 imports
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import InMemorySaver

# Gradio
import gradio as gr

# 환경 변수 설정
from dotenv import load_dotenv
load_dotenv()

# # Langfuse 추적 활성화
# os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY", "")
# os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY", "")
# os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
# os.environ["LANGCHAIN_TRACING_V2"] = "false"
# os.environ["LANGCHAIN_ENDPOINT"] = ""

# ============================================================================
# 1. 상태 정의
# ============================================================================

class InfrastructureState(TypedDict):
    """인프라 모니터링 상태"""
    # 입력 데이터
    user_query: str                    # 사용자 질문
    alert_data: Optional[Dict]         # 알람 데이터
    date_range: Optional[str]          # 분석 기간
    
    # 라우팅 정보
    route: Optional[str]               # 라우팅 결과
    
    # 수집된 데이터
    hadoop_metrics: List[Dict]         # 하둡 메트릭
    presto_metrics: List[Dict]         # Presto 메트릭
    jenkins_metrics: List[Dict]        # Jenkins 메트릭
    grafana_metrics: List[Dict]        # Grafana 메트릭
    
    # 분석 결과
    daily_report: Optional[str]        # 일일 리포트
    alert_analysis: Optional[str]      # 알람 분석 결과
    spark_jobs: List[Dict]             # Spark 잡 정보
    recommendations: List[str]         # 권장사항
    
    # 에이전트 응답
    infra_engineer_response: Optional[str]
    data_engineer_response: Optional[str]
    admin_guide_response: Optional[str]
    
    # 최종 결과
    final_answer: Optional[str]

# ============================================================================
# 2. 도구 정의 (Tool Functions)
# ============================================================================

@tool
def get_cluster_metrics() -> Dict:
    """클러스터 시스템 리소스 사용량을 가져옵니다."""
    # 데모용 가상 데이터
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_usage": {
            "overall": 75.5,
            "node1": 82.3,
            "node2": 68.7,
            "node3": 79.1
        },
        "memory_usage": {
            "overall": 78.2,
            "node1": 85.6,
            "node2": 72.4,
            "node3": 76.5
        },
        "disk_usage": {
            "overall": 65.8,
            "node1": 71.2,
            "node2": 62.3,
            "node3": 63.9
        },
        "network_usage": {
            "overall": 45.3,
            "node1": 52.1,
            "node2": 41.8,
            "node3": 42.0
        }
    }

@tool
def get_spark_jobs() -> List[Dict]:
    """현재 실행 중인 Spark 잡 정보를 가져옵니다."""
    # 데모용 가상 데이터
    return [
        {
            "job_id": "spark_001",
            "duration": "2h 15m",
            "cpu_usage": 85.2,
            "memory_usage": 78.5
        },
        {
            "job_id": "spark_002",
            "duration": "3h 45m",
            "cpu_usage": 92.1,
            "memory_usage": 89.3
        },
        {
            "job_id": "spark_003",
            "duration": "1h 30m",
            "cpu_usage": 45.7,
            "memory_usage": 52.1
        }
    ]

@tool
def kill_spark_job(job_id: str) -> Dict:
    """지정된 Spark 잡을 종료합니다."""
    # 데모용 가상 응답
    return {
        "job_id": job_id,
        "action": "KILLED",
        "timestamp": datetime.now().isoformat(),
        "status": "SUCCESS",
        "message": f"Job {job_id} has been successfully terminated."
    }

# 운영 매뉴얼 벡터 데이터베이스 초기화
def initialize_operation_manual_db():
    """운영 매뉴얼을 Chroma 벡터 데이터베이스에 로드"""
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # 텍스트 파일 로드
    loader = TextLoader("data/operation_manual.txt", encoding="utf-8")
    documents = loader.load()
    
    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    
    # Chroma 벡터 데이터베이스 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="operation_manual",
        persist_directory="./chroma_db"
    )
    
    return vectorstore

# 전역 변수로 벡터 데이터베이스 저장
operation_manual_vectorstore = None

@tool
def search_operation_manual(query: str) -> List[Document]:
    """운영 매뉴얼에서 관련 정보를 벡터 유사도 검색으로 찾습니다."""
    global operation_manual_vectorstore
    
    # 벡터 데이터베이스가 없으면 초기화
    if operation_manual_vectorstore is None:
        print("운영 매뉴얼 벡터 데이터베이스 초기화 중...")
        operation_manual_vectorstore = initialize_operation_manual_db()
    
    # 유사도 검색 수행
    docs = operation_manual_vectorstore.similarity_search(query, k=3)
    
    if docs:
        return docs
    else:
        return [Document(
            page_content="관련 운영 매뉴얼 정보를 찾을 수 없습니다.",
            metadata={"source": "yg_operation_manual.txt"}
        )]

# 운영 이력 벡터 데이터베이스 초기화
def initialize_operation_history_db():
    """운영 이력을 Chroma 벡터 데이터베이스에 로드"""
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # 텍스트 파일 로드
    loader = TextLoader("data/operation_history.txt", encoding="utf-8")
    documents = loader.load()
    
    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    
    # Chroma 벡터 데이터베이스 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="operation_history",
        persist_directory="./chroma_db"
    )
    
    return vectorstore

# 전역 변수로 벡터 데이터베이스 저장
operation_history_vectorstore = None

@tool
def search_operation_history(query: str) -> List[Document]:
    """운영 이력에서 관련 정보를 벡터 유사도 검색으로 찾습니다."""
    global operation_history_vectorstore
    
    # 벡터 데이터베이스가 없으면 초기화
    if operation_history_vectorstore is None:
        print("운영 이력 벡터 데이터베이스 초기화 중...")
        operation_history_vectorstore = initialize_operation_history_db()
    
    # 유사도 검색 수행
    docs = operation_history_vectorstore.similarity_search(query, k=3)
    
    if docs:
        return docs
    else:
        return [Document(
            page_content="관련 운영 이력을 찾을 수 없습니다.",
            metadata={"source": "yg_operation_history.txt"}
        )]

@tool
def search_internet_info(query: str) -> List[Document]:
    """Tavily를 사용하여 인터넷에서 관련 정보를 검색합니다."""
    try:
        from tavily import TavilyClient
        
        # Tavily 클라이언트 초기화
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            error_msg = "TAVILY_API_KEY가 설정되지 않았습니다. 인터넷 검색을 수행할 수 없습니다."
            print(error_msg)
            return [Document(
                page_content=error_msg,
                metadata={"source": "error", "error_type": "missing_api_key"}
            )]
        
        client = TavilyClient(api_key=tavily_api_key)
        
        # 검색 쿼리 개선 (데이터 인프라 관련 키워드 추가)
        enhanced_query = f"{query} data infrastructure monitoring best practices"
        
        # Tavily 검색 실행
        search_result = client.search(
            query=enhanced_query,
            search_depth="advanced",
            max_results=5,
            include_domains=["apache.org", "databricks.com", "cloudera.com", "hortonworks.com", "stackoverflow.com", "medium.com", "towardsdatascience.com"]
        )
        
        # 검색 결과를 Document 형태로 변환
        documents = []
        for result in search_result.get("results", []):
            content = result.get("content", "")
            url = result.get("url", "")
            title = result.get("title", "")
            
            if content:
                # 제목과 내용을 결합
                full_content = f"제목: {title}\n\n내용: {content}"
                documents.append(Document(
                    page_content=full_content,
                    metadata={
                        "source": "tavily_search",
                        "url": url,
                        "title": title
                    }
                ))
        
        if documents:
            return documents
        else:
            error_msg = "Tavily 검색 결과가 없습니다. 인터넷 검색을 수행할 수 없습니다."
            print(error_msg)
            return [Document(
                page_content=error_msg,
                metadata={"source": "error", "error_type": "no_search_results"}
            )]
            
    except Exception as e:
        error_msg = f"Tavily 검색 중 오류 발생: {e}. 인터넷 검색을 수행할 수 없습니다."
        print(error_msg)
        return [Document(
            page_content=error_msg,
            metadata={"source": "error", "error_type": "search_error", "error_details": str(e)}
        )]

# 도구 목록
tools = [
    get_cluster_metrics,
    get_spark_jobs,
    kill_spark_job,
    search_operation_manual,
    search_operation_history,
    search_internet_info
]

# ============================================================================
# 3. LLM 모델 설정
# ============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ============================================================================
# 4. 에이전트 구현
# ============================================================================

def infra_engineer_agent(state: InfrastructureState) -> InfrastructureState:
    """인프라 엔지니어 에이전트 - 클러스터 가동율 및 시스템 현황 담당"""
    print("--- 인프라 엔지니어 에이전트 시작 ---")
    
    query = state["user_query"]
    
    # 클러스터 메트릭 가져오기 (Spark 잡 로직과 동일하게 변경)
    ai_msg = llm_with_tools.invoke(query)

    tool_call = ai_msg.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    for tool in tools:
        if tool_name == tool.name:
            metrics = tool.invoke(tool_args)
    
    # 분석 프롬프트
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 데이터 인프라 엔지니어입니다. 클러스터의 시스템 리소스 사용 현황을 분석하고 사용자 질문에 답변해주세요.
        
        분석해야 할 항목:
        1. CPU 사용률 (전체 및 개별 노드)
        2. 메모리 사용률 (전체 및 개별 노드)
        3. 디스크 사용률 (전체 및 개별 노드)
        4. 네트워크 사용률 (전체 및 개별 노드)
        
        답변 형식:
        - 현재 클러스터 상태 요약
        - 각 노드별 상세 현황
        - 주의가 필요한 부분
        - 권장사항"""),
        ("human", "사용자 질문: {query}\n\n클러스터 메트릭: {metrics}")
    ])
    
    response = llm.invoke(analysis_prompt.format(
        query=query,
        metrics=json.dumps(metrics, indent=2, ensure_ascii=False)
    ))
    
    return {
        "hadoop_metrics": [metrics],
        "infra_engineer_response": response.content,
        "data_engineer_response": state.get("data_engineer_response", ""),
        "admin_guide_response": state.get("admin_guide_response", "")
    }

def data_engineer_agent(state: InfrastructureState) -> InfrastructureState:
    """데이터 엔지니어 에이전트 - Spark 잡 관리 담당"""
    print("--- 데이터 엔지니어 에이전트 시작 ---")
    
    query = state["user_query"]
    
    # Spark 잡 정보 가져오기
    ai_msg = llm_with_tools.invoke(query)

    tool_call = ai_msg.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    for tool in tools:
        if tool_name == tool.name:
            spark_jobs = tool.invoke(tool_args)
    
    # 상위 3개 잡 추출 (CPU 사용률 기준)
    top_jobs = sorted(spark_jobs, key=lambda x: x["cpu_usage"], reverse=True)[:3]
    
    # 분석 프롬프트
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 데이터 엔지니어입니다. Spark 잡 현황을 분석하고 사용자 질문에 답변해주세요.
        
        분석해야 할 항목:
        1. 현재 실행 중인 Spark 잡 목록
        2. 리소스 사용량이 높은 잡 식별
        3. 수행 시간이 긴 잡 확인
        4. 잡 종료 권장사항
        
        답변 형식:
        - Spark 잡 현황 요약
        - 상위 3개 잡 상세 정보
        - 리소스 사용량 분석
        - 잡 관리 권장사항"""),
        ("human", "사용자 질문: {query}\n\nSpark 잡 정보: {spark_jobs}")
    ])
    
    response = llm.invoke(analysis_prompt.format(
        query=query,
        spark_jobs=json.dumps(top_jobs, indent=2, ensure_ascii=False)
    ))
    
    return {
        "spark_jobs": top_jobs,
        "data_engineer_response": response.content,
        "infra_engineer_response": state.get("infra_engineer_response", ""),
        "admin_guide_response": state.get("admin_guide_response", "")
    }

def admin_guide_agent(state: InfrastructureState) -> InfrastructureState:
    """관리자 가이드 에이전트 - 운영 매뉴얼 및 이력 검색 담당"""
    print("--- 관리자 가이드 에이전트 시작 ---")
    
    query = state["user_query"]
    
    # 운영 매뉴얼, 운영 이력, 인터넷 정보 검색을 각각 tool 호출 방식으로 처리 (ai_msg 중복 방지)
    manual_tool_call = llm_with_tools.invoke(f"운영 매뉴얼에서 답을 찾아줘: {query}")
    manual_tool_name = manual_tool_call.tool_calls[0]["name"]
    manual_tool_args = manual_tool_call.tool_calls[0]["args"]
    manual_docs = None
    for tool in tools:
        if manual_tool_name == tool.name:
            manual_docs = tool.invoke(manual_tool_args)

    history_tool_call = llm_with_tools.invoke(f"운영 이력에서 답을 찾아줘: {query}")
    history_tool_name = history_tool_call.tool_calls[0]["name"]
    history_tool_args = history_tool_call.tool_calls[0]["args"]
    history_docs = None
    for tool in tools:
        if history_tool_name == tool.name:
            history_docs = tool.invoke(history_tool_args)

    internet_tool_call = llm_with_tools.invoke(f"인터넷에서 답을 찾아줘: {query}")
    internet_tool_name = internet_tool_call.tool_calls[0]["name"]
    internet_tool_args = internet_tool_call.tool_calls[0]["args"]
    internet_docs = None
    for tool in tools:
        if internet_tool_name == tool.name:
            internet_docs = tool.invoke(internet_tool_args)
    
    # 분석 프롬프트
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 데이터 인프라 관리자입니다. 운영 매뉴얼, 이전 운영 이력, 인터넷 정보를 종합하여 사용자 질문에 답변해주세요.
        
        참고 자료:
        1. 운영 매뉴얼: 표준 운영 절차 및 가이드라인
        2. 운영 이력: 이전에 발생한 유사 상황과 해결 방법
        3. 인터넷 정보: 최신 기술 정보 및 모범 사례
        
        답변 형식:
        - 운영 매뉴얼 기반 표준 절차
        - 이전 사례 분석
        - 최신 정보 및 권장사항
        - 종합적인 해결 방안"""),
        ("human", """사용자 질문: {query}

운영 매뉴얼:
{manual_docs}

운영 이력:
{history_docs}

인터넷 정보:
{internet_docs}""")
    ])
    
    response = llm.invoke(analysis_prompt.format(
        query=query,
        manual_docs="\n".join([doc.page_content for doc in manual_docs]),
        history_docs="\n".join([doc.page_content for doc in history_docs]),
        internet_docs="\n".join([doc.page_content for doc in internet_docs])
    ))
    
    return {
        "admin_guide_response": response.content,
        "infra_engineer_response": state.get("infra_engineer_response", ""),
        "data_engineer_response": state.get("data_engineer_response", "")
    }

def route_query(state: InfrastructureState) -> InfrastructureState:
    """사용자 질문을 분석하여 적절한 에이전트로 라우팅"""
    print("--- 질문 라우팅 분석 ---")
    
    query = state["user_query"].lower()
    
    # 인프라 관련 키워드
    infra_keywords = [
        "클러스터", "cpu", "메모리", "디스크", "네트워크", "리소스", "가동율", "사용률",
        "시스템", "노드", "서버", "성능", "부하", "부족", "높음", "낮음",
        "모니터링", "메트릭", "현황", "상태", "확인", "점검"
    ]
    
    # 데이터/데이터플랫폼 관련 키워드
    data_keywords = [
        "spark", "hadoop", "hdfs", "yarn", "presto", "hive", "kafka", "flink",
        "잡", "job", "파이프라인", "pipeline", "etl", "데이터", "처리",
        "kill", "종료", "실행", "스케줄", "큐", "메모리", "cpu 사용률"
    ]
    
    # 운영 매뉴얼/이력 관련 키워드
    admin_keywords = [
        "매뉴얼", "운영", "절차", "가이드", "방법", "조치", "대응", "해결",
        "이력", "경험", "사례", "전례", "이전", "과거", "학습점",
        "어떻게", "무엇을", "왜", "언제", "어디서", "누가"
    ]
    
    # 키워드 매칭 점수 계산
    infra_score = sum(1 for keyword in infra_keywords if keyword in query)
    data_score = sum(1 for keyword in data_keywords if keyword in query)
    admin_score = sum(1 for keyword in admin_keywords if keyword in query)
    
    print(f"라우팅 점수 - 인프라: {infra_score}, 데이터: {data_score}, 관리자: {admin_score}")
    
    # 가장 높은 점수의 에이전트로 라우팅
    if data_score > infra_score and data_score > admin_score:
        print("→ 데이터 엔지니어 에이전트로 라우팅")
        return {"route": "data_engineer"}
    elif admin_score > infra_score and admin_score > data_score:
        print("→ 관리자 가이드 에이전트로 라우팅")
        return {"route": "admin_guide"}
    else:
        print("→ 인프라 엔지니어 에이전트로 라우팅")
        return {"route": "infra_engineer"}

def generate_final_answer(state: InfrastructureState) -> InfrastructureState:
    """최종 답변 생성"""
    print("--- 최종 답변 생성 ---")
    
    query = state["user_query"]
    infra_response = state.get("infra_engineer_response", "")
    data_response = state.get("data_engineer_response", "")
    admin_response = state.get("admin_guide_response", "")
    
    # 최종 답변 생성 프롬프트
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 데이터 인프라 모니터링 전문가입니다. 처리된 에이전트의 응답을 바탕으로 사용자에게 명확하고 실용적인 답변을 제공해주세요.
        
        답변 구조:
        1. 현재 상황 요약
        2. 상세 분석 결과
        3. 구체적인 조치 방안
        4. 예방 및 모니터링 권장사항
        
        각 에이전트의 전문 분야:
        - 인프라 엔지니어: 시스템 리소스 현황 및 클러스터 상태
        - 데이터 엔지니어: Spark 잡 관리 및 최적화
        - 관리자 가이드: 운영 절차 및 이전 사례"""),
        ("human", """사용자 질문: {query}

인프라 엔지니어 응답:
{infra_response}

데이터 엔지니어 응답:
{data_response}

관리자 가이드 응답:
{admin_response}

위 정보를 종합하여 최종 답변을 작성해주세요.""")
    ])
    
    response = llm.invoke(final_prompt.format(
        query=query,
        infra_response=infra_response,
        data_response=data_response,
        admin_response=admin_response
    ))
    
    return {
        "final_answer": response.content
    }

# ============================================================================
# 5. LangGraph 워크플로우 구성
# ============================================================================

def create_infrastructure_monitoring_workflow():
    """인프라 모니터링 워크플로우 생성"""
    
    # 그래프 생성
    workflow = StateGraph(InfrastructureState)
    
    # 노드 추가
    workflow.add_node("route", route_query)
    workflow.add_node("infra_engineer", infra_engineer_agent)
    workflow.add_node("data_engineer", data_engineer_agent)
    workflow.add_node("admin_guide", admin_guide_agent)
    workflow.add_node("generate_answer", generate_final_answer)
    
    # 시작점에서 라우팅 노드로
    workflow.add_edge(START, "route")
    
    # 조건부 엣지 추가 (라우팅 결과에 따라)
    workflow.add_conditional_edges(
        "route",
        lambda x: x["route"],
        {
            "infra_engineer": "infra_engineer",
            "data_engineer": "data_engineer", 
            "admin_guide": "admin_guide"
        }
    )
    
    # 각 에이전트에서 최종 답변 생성으로
    workflow.add_edge("infra_engineer", "generate_answer")
    workflow.add_edge("data_engineer", "generate_answer")
    workflow.add_edge("admin_guide", "generate_answer")
    
    # 최종 답변에서 종료
    workflow.add_edge("generate_answer", END)
    
    # 그래프 컴파일
    return workflow.compile()

# ============================================================================
# 6. Gradio 인터페이스
# ============================================================================

def process_query(query: str) -> str:
    """사용자 질문을 처리하고 답변을 반환"""
    try:
        # 워크플로우 생성
        workflow = create_infrastructure_monitoring_workflow()
        
        # 워크플로우 실행
        result = workflow.invoke({
            "user_query": query
        })
        
        return result.get("final_answer", "답변을 생성할 수 없습니다.")
        
    except Exception as e:
        return f"오류가 발생했습니다: {str(e)}"

def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    
    with gr.Blocks(title="데이터 인프라 모니터링 서비스", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🏗️ LLM 기반 데이터 인프라 모니터링 서비스")
        gr.Markdown("클러스터 CPU 사용율이 높다는 알람을 받은 후 챗봇을 통해 인프라 상황을 분석하고 조치할 수 있습니다.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 질문 입력")
                query_input = gr.Textbox(
                    label="질문을 입력하세요",
                    placeholder="예: 현재 클러스터 가동율 알려줘?",
                    lines=3
                )
                
                submit_btn = gr.Button("🔍 분석 시작", variant="primary")
                
                # 예시 질문들
                gr.Markdown("### 💡 예시 질문")
                example_queries = [
                    "현재 클러스터 가동율 알려줘?",
                    "클러스터 가동율이 높을때 어떻게 해야되?",
                    "리소스를 많이 사용하는 spark 잡 3개 추출해줘?",
                    "이 중에서 상위 2개 spark 잡 Kill 해줘?",
                    "클러스터 사용율이 떨어졌는지 확인해줘?",
                    "운영 매뉴얼에서 장애 조치 방법 찾아줘.",
                    "이전에 비슷한 장애가 있었는지 운영 이력에서 검색해줘."
                ]
                
                for example in example_queries:
                    gr.Button(example, size="sm").click(
                        lambda q=example: q,
                        outputs=query_input
                    )
            
            with gr.Column(scale=2):
                gr.Markdown("### 🤖 분석 결과")
                result_output = gr.Markdown(
                    label="분석 결과",
                    value="질문을 입력하고 분석을 시작해주세요."
                )
        
        # 이벤트 연결
        submit_btn.click(
            fn=process_query,
            inputs=query_input,
            outputs=result_output
        )
        
        # Enter 키로도 제출 가능
        query_input.submit(
            fn=process_query,
            inputs=query_input,
            outputs=result_output
        )
    
    return interface

# ============================================================================
# 7. 메인 실행 함수
# ============================================================================

def main():
    """메인 실행 함수"""
    print("🚀 데이터 인프라 모니터링 서비스를 시작합니다...")
    
    # Gradio 인터페이스 생성
    interface = create_gradio_interface()
    
    # 서버 시작
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    main()