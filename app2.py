import os
import base64
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

st.title("🖼️ 이미지 설명 챗봇 (Azure OpenAI)")

azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
azure_oai_key = os.getenv("AZURE_OAI_KEY")
azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT", "gpt-4o-mini")

if not azure_oai_endpoint or not azure_oai_key or not azure_oai_deployment:
    st.error("환경변수(AZURE_OAI_ENDPOINT / AZURE_OAI_KEY / AZURE_OAI_DEPLOYMENT)를 확인하세요.")
    st.stop()

client = AzureOpenAI(
    azure_endpoint=azure_oai_endpoint,   # ✅ 변수명 일치
    api_key=azure_oai_key,
    api_version="2025-01-01-preview",
)

uploaded = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg", "webp"])

prompt = st.text_input("질문", value="이 사진 설명해봐.")

if uploaded is not None:
    st.image(uploaded, caption="업로드된 이미지", use_container_width=True)

if st.button("전송") and uploaded is not None:
    img_bytes = uploaded.getvalue()
    mime = uploaded.type or "application/octet-stream"  # ✅ webp면 image/webp
    b64 = base64.b64encode(img_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"              # ✅ MIME 자동 반영 (중요)

    messages = [
        {"role": "system", "content": "사용자가 정보를 찾는 데 도움이 되는 AI 도우미입니다."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ]

    try:
        completion = client.chat.completions.create(
            model=azure_oai_deployment,
            messages=messages,
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
        )
        st.subheader("결과")
        st.write(completion.choices[0].message.content)
    except Exception as e:
        st.error("호출 실패")
        st.exception(e)
