import streamlit as st
import pandas as pd
import openai
from dotenv import load_dotenv
import os
from datetime import datetime
import pdfplumber # PDF 처리를 위한 라이브러리

# --- 초기 설정 ---

# .env 파일에서 환경 변수(API 키)를 로드합니다.
load_dotenv()

# --- 핵심 함수 정의 ---

@st.cache_data # Streamlit의 캐시 기능을 사용하여 중복 작업을 피합니다.
def process_uploaded_file(uploaded_file):
    """업로드된 파일을 처리하여 DataFrame 또는 텍스트를 반환합니다."""
    if uploaded_file is None:
        return None
    
    # 파일 이름에서 확장자 추출
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    if file_extension == ".csv":
        # CSV 파일 처리
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
            return None
    elif file_extension == ".pdf":
        # PDF 파일 처리
        try:
            text_content = ""
            with pdfplumber.open(uploaded_file) as pdf:
                # PDF의 모든 페이지를 순회하며 텍스트 추출
                for page in pdf.pages:
                    text_content += page.extract_text() + "\n"
            return text_content
        except Exception as e:
            st.error(f"PDF 파일을 읽는 중 오류가 발생했습니다: {e}")
            return None
    else:
        st.error("지원하지 않는 파일 형식입니다. CSV 또는 PDF 파일을 업로드해주세요.")
        return None

def get_date_from_query(query):
    """사용자의 질문(query)에서 날짜를 추출하기 위해 OpenAI API를 호출하는 함수입니다."""
    try:
        today_str = datetime.now().strftime("%Y년 %m월 %d일")
        system_prompt = f"""
        당신은 사용자의 질문에서 날짜를 추출하는 역할을 맡았습니다.
        오늘 날짜는 {today_str}입니다.
        사용자의 질문을 분석해서 'YYYY-MM-DD' 형식의 날짜를 반환해야 합니다.
        예를 들어:
        - 사용자가 "오늘"이라고 하면, 오늘의 날짜를 반환합니다.
        - 사용자가 "내일"이라고 하면, 내일의 날짜를 계산하여 반환합니다.
        - 사용자가 "8월 8일"이라고 하면, 현재 연도를 기준으로 "2025-08-08"을 반환합니다.
        - 만약 질문에서 날짜 관련 표현을 찾을 수 없다면, "None"을 반환합니다.
        
        출력은 반드시 'YYYY-MM-DD' 형식의 날짜 또는 "None" 문자열이어야 합니다. 다른 설명은 절대 추가하지 마세요.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.0,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"날짜 추출 중 OpenAI API 호출에 실패했습니다: {e}")
        return "None"

def get_menu_response(query, context_data):
    """
    RAG 파이프라인 전체를 실행하여 사용자 질문에 대한 최종 답변을 생성하는 함수입니다.
    이제 context_data는 DataFrame(CSV) 또는 str(PDF)일 수 있습니다.
    """
    
    # 1. 검색 (Retrieval): 사용자 질문에서 날짜 추출
    extracted_date_str = get_date_from_query(query)
    
    if extracted_date_str == "None":
        return "날짜와 관련된 질문을 해주시겠어요? 예를 들어 '오늘 급식 뭐야?' 와 같이 물어보실 수 있어요."

    # 2. 증강 (Augmentation) 및 3. 생성 (Generation)
    # 데이터 타입에 따라 다른 프롬프트를 사용하여 LLM에 요청합니다.

    if isinstance(context_data, pd.DataFrame):
        # --- CSV (DataFrame) 처리 로직 ---
        menu_info = context_data[context_data['date'] == extracted_date_str]

        if menu_info.empty:
            return f"{extracted_date_str}에 대한 급식 정보가 등록되어 있지 않아요. 다른 날짜를 물어봐 주시겠어요?"

        retrieved_menu = menu_info.iloc[0]['menu']
        retrieved_day = menu_info.iloc[0]['day']
        
        system_prompt = "당신은 학교 급식 메뉴를 친절하게 안내하는 챗봇입니다. 주어진 '검색된 급식 정보'를 바탕으로 답변해야 합니다."
        augmented_prompt = f"""
        [검색된 급식 정보]
        - 날짜: {extracted_date_str}
        - 요일: {retrieved_day}
        - 메뉴: {retrieved_menu}

        위 정보를 바탕으로 아래 사용자 질문에 대해 상냥하고 명랑한 말투로 답변해주세요.
        [사용자 질문]
        {query}
        """
    elif isinstance(context_data, str):
        # --- PDF (텍스트) 처리 로직 ---
        system_prompt = "당신은 제공된 'PDF 문서 내용' 전체를 참고하여 사용자의 질문에 답변하는 급식 안내 챗봇입니다."
        augmented_prompt = f"""
        [PDF 문서 내용]
        {context_data}

        위 PDF 문서 내용을 바탕으로, '{extracted_date_str}' 날짜의 급식 메뉴가 무엇인지 찾아서 아래 사용자 질문에 답변해주세요.
        만약 해당 날짜의 정보를 찾을 수 없다면, 정보가 없다고 솔직하게 답변해야 합니다.
        [사용자 질문]
        {query}
        """
    else:
        return "처리할 수 없는 데이터 형식입니다."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message['content']
    except Exception as e:
        st.error(f"답변 생성 중 OpenAI API 호출에 실패했습니다: {e}")
        return "죄송합니다, 답변을 생성하는 중에 문제가 발생했어요."

# --- Streamlit UI 설정 ---

st.set_page_config(page_title="RAG 급식 챗봇", page_icon="🍚")
st.title("📄 RAG 급식 챗봇 (파일 업로드)")
st.write("CSV 또는 PDF 형식의 급식 메뉴 파일을 업로드하여 챗봇과 대화해보세요.")

# 사이드바 설정
with st.sidebar:
    st.header("설정")
    api_key = st.text_input(
        "OpenAI API Key", 
        type="password", 
        help="OpenAI API 키를 입력하세요."
    )
    
    st.divider()

    # 파일 업로더
    uploaded_file = st.file_uploader(
        "급식 메뉴 파일(CSV 또는 PDF)을 업로드하세요.",
        type=["csv", "pdf"]
    )

# API 키 설정
if api_key:
    openai.api_key = api_key
else:
    env_api_key = os.getenv("OPENAI_API_KEY")
    if env_api_key:
        openai.api_key = env_api_key
    else:
        st.warning("사이드바에 OpenAI API 키를 입력해주세요.")
        st.stop()

# 파일이 업로드 되었는지 확인
if uploaded_file is not None:
    # 파일을 처리하고 결과를 세션 상태에 저장하여 재실행 시에도 유지되도록 함
    st.session_state.context_data = process_uploaded_file(uploaded_file)
    
    if st.session_state.context_data is not None:
        st.success(f"'{uploaded_file.name}' 파일이 성공적으로 처리되었습니다. 이제 질문을 시작하세요!")
        
        # 채팅 기록 초기화
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 이전 채팅 기록 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 사용자 입력 처리
        if prompt := st.chat_input("오늘 급식 뭐야?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner('답변을 생성하는 중입니다...'):
                # 세션에 저장된 context 데이터를 사용
                response = get_menu_response(prompt, st.session_state.context_data)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
else:
    st.info("시작하려면 사이드바에서 급식 메뉴 파일을 업로드해주세요.")



### 사용 방법

#1.  **라이브러리 추가 설치:**
#    PDF 파일을 읽기 위해 `pdfplumber` 라이브러리를 추가로 설치해야 합니다. 터미널에 아래 명령어를 입력하세요.
#    ```bash
#    pip install pdfplumber
#    ```
#    (기존 라이브러리도 설치되어 있어야 합니다: `pip install streamlit openai pandas python-dotenv`)

#2.  **파일 실행:**
#    위 코드를 `app.py`로 저장하고, 터미널에서 아래 명령어를 실행하세요.
#    ```bash
#    streamlit run app.py
#    ```

#3.  **챗봇 사용:**#
    #* 웹 브라우저에 챗봇이 나타나면, 먼저 **사이드바에 OpenAI API 키를 입력**하세요.
#    * 그 다음, **사이드바의 파일 업로더를 사용**하여 가지고 있는 급식 메뉴 파일(CSV 또는 PDF)을 업로드합니다.
#    * 파일이 성공적으로 처리되었다는 메시지가 나타나면, 채팅창에 질문을 입력하여 대화를 시작할 수 있습
#'''
