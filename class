import streamlit as st
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI 챗봇 & 이미지 생성기",
    page_icon="🎨",
    layout="centered"
)

# --- 사이드바 ---
with st.sidebar:
    st.title("🤖 AI 챗봇 & 이미지 생성기")
    st.markdown("---")
    
    # 챗봇 모드 선택 기능 추가
    chat_mode = st.selectbox(
        "원하는 챗봇을 선택하세요:",
        ("어린이 대화 챗봇", "이미지 생성 챗봇")
    )
    st.markdown("---")

    # OpenAI API 키 입력
    openai_api_key = st.text_input(
        "OpenAI API Key를 입력해주세요.", 
        type="password"
    )

    # 선택된 모드에 따라 다른 안내 메시지 표시
    if chat_mode == "어린이 대화 챗봇":
        st.markdown("""
        **사용 안내 (대화 챗봇):**
        1. OpenAI API 키를 발급받아 위에 입력하세요.
        2. 아래 채팅창에 질문을 입력해보세요.
        3. 챗봇 '바른이'가 비속어를 감지하면, 올바른 표현을 사용하도록 유도할 거예요.
        """)
    elif chat_mode == "이미지 생성 챗봇":
        st.markdown("""
        **사용 안내 (이미지 생성):**
        1. OpenAI API 키를 발급받아 위에 입력하세요.
        2. 아래 채팅창에 만들고 싶은 이미지를 **자세히** 설명해주세요. (예: '푸른 하늘을 나는 우주비행사 고양이')
        3. AI가 설명을 바탕으로 멋진 이미지를 만들어 줄 거예요!
        """)

# --- 세션 상태(Session State) 초기화 ---
# 각 챗봇의 대화 기록을 별도로 관리하기 위함

# 1. 어린이 대화 챗봇의 대화 기록
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="안녕하세요! 저는 어린이들의 친구, 바른이라고 해요. 궁금한 것을 물어보세요! 👋")
    ]

# 2. 이미지 생성 챗봇의 대화 기록
if "image_messages" not in st.session_state:
    st.session_state.image_messages = []


# --- 메인 화면 ---
# 선택된 챗봇 모드에 따라 다른 UI와 기능을 렌더링

## 1. 어린이 대화 챗봇 모드
if chat_mode == "어린이 대화 챗봇":
    st.title("어린이 비속어 방지 챗봇 👧👦")
    st.markdown("아이들이 올바른 언어 습관을 기를 수 있도록 도와주는 친구예요.")
    st.markdown("---")

    # 시스템 메시지 (챗봇의 역할 정의)
    SYSTEM_MESSAGE = """
    너는 어린이들을 위한 챗봇이야. 이름은 '바른이'이고, 항상 친절하고 다정하게 존댓말로 대답해야 해. 
    너의 가장 중요한 임무는 아이들이 비속어나 나쁜 말을 사용하지 않도록 돕는 거야.

    만약 사용자가 비속어, 욕설, 또는 공격적인 말을 사용하면, 절대로 그 말을 따라하거나 화내지 마.
    대신, 아래와 같이 부드럽게 지적하고 올바른 말을 사용하도록 친절하게 유도해야 해.

    예시:
    - 사용자: "야, 이 바보야"
    - 너의 답변: "친구야, '바보'라는 말은 다른 사람의 마음을 아프게 할 수 있어. 대신 '친절하게 말해줄래?'라고 이야기하는 건 어떨까? 😊"
    - 사용자: "이거 진짜 짜증나."
    - 너의 답변: "그렇구나, 많이 속상했겠다. '짜증난다'는 말 대신 '이것 때문에 조금 속상해'라고 표현하면 마음이 더 잘 전달될 거야."

    항상 긍정적이고 교육적인 태도를 유지해줘.
    """

    # 이전 대화 기록 표시
    for message in st.session_state.messages:
        if isinstance(message, AIMessage):
            with st.chat_message("AI", avatar="🤖"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human", avatar="🧑"):
                st.markdown(message.content)

    # 사용자 입력 처리
    if user_input := st.chat_input("메시지를 입력하세요..."):
        if not openai_api_key:
            st.error("먼저 사이드바에 OpenAI API Key를 입력해주세요! 🔑")
            st.stop()
        
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.chat_message("Human", avatar="🧑"):
            st.markdown(user_input)
        
        with st.spinner("바른이가 생각하고 있어요... 🤔"):
            try:
                chat = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=openai_api_key)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", SYSTEM_MESSAGE),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ])
                chain = prompt | chat
                response = chain.invoke({"chat_history": st.session_state.messages, "input": user_input})
                ai_response = response.content
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
                ai_response = "미안해요, 지금은 답변하기 어려워요. API 키가 올바른지 확인해주세요."

        st.session_state.messages.append(AIMessage(content=ai_response))
        with st.chat_message("AI", avatar="🤖"):
            st.markdown(ai_response)


## 2. 이미지 생성 챗봇 모드
elif chat_mode == "이미지 생성 챗봇":
    st.title("AI 이미지 생성기 🎨")
    st.markdown("만들고 싶은 이미지를 텍스트로 설명해주세요.")
    st.markdown("---")

    # 이전 대화(이미지 생성 이력) 표시
    for msg in st.session_state.image_messages:
        if msg["role"] == "user":
            with st.chat_message("Human", avatar="🧑"):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("AI", avatar="🎨"):
                st.image(msg["content"], caption="AI가 생성한 이미지")

    # 사용자 입력 처리
    if user_prompt := st.chat_input("만들고 싶은 이미지를 설명해주세요..."):
        if not openai_api_key:
            st.error("먼저 사이드바에 OpenAI API Key를 입력해주세요! 🔑")
            st.stop()

        # 사용자 프롬프트를 기록에 추가하고 화면에 표시
        st.session_state.image_messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("Human", avatar="🧑"):
            st.markdown(user_prompt)

        # 이미지 생성
        with st.spinner("멋진 이미지를 만들고 있어요... ✨"):
            try:
                # OpenAI 클라이언트 초기화
                client = OpenAI(api_key=openai_api_key)
                
                # DALL-E 3 모델을 사용하여 이미지 생성 요청
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=user_prompt,
                    size="1024x1024", # 이미지 크기
                    quality="standard", # 이미지 품질
                    n=1, # 생성할 이미지 개수
                )
                
                # 생성된 이미지의 URL 추출
                image_url = response.data[0].url
                
                # 생성된 이미지를 기록에 추가하고 화면에 표시
                st.session_state.image_messages.append({"role": "assistant", "content": image_url})
                with st.chat_message("AI", avatar="🎨"):
                    st.image(image_url, caption=f"'{user_prompt}'에 대한 이미지")

            except Exception as e:
                st.error(f"이미지 생성 중 오류가 발생했습니다: {e}")
                st.error("API 키가 유효한지, 또는 프롬프트에 부적절한 내용이 포함되지 않았는지 확인해주세요.")
