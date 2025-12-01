import os
import base64
import hashlib
import streamlit as st
from openai import AzureOpenAI

# -------------------------
# (선택) 로컬 개발용 .env 지원: 배포에서 없어도 안 터지게
# -------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

st.set_page_config(page_title="이미지 설명 챗봇", page_icon="🖼️", layout="centered")
st.title("🖼️ 이미지 설명 챗봇 (Azure OpenAI)")

# -------------------------
# 환경변수(또는 Streamlit Secrets)에서 읽기
# -------------------------
azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
azure_oai_key = os.getenv("AZURE_OAI_KEY")
azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT", "gpt-4o-mini")

if not azure_oai_endpoint or not azure_oai_key or not azure_oai_deployment:
    st.error("환경변수(AZURE_OAI_ENDPOINT / AZURE_OAI_KEY / AZURE_OAI_DEPLOYMENT)를 확인하세요.")
    st.stop()

client = AzureOpenAI(
    azure_endpoint=azure_oai_endpoint,
    api_key=azure_oai_key,
    api_version="2025-01-01-preview",
)

# -------------------------
# 프롬프트 템플릿 (가볍게, 티 많이 남게)
# -------------------------
SYSTEM_GENERAL = (
    "너는 사용자가 이미지를 이해하도록 돕는 친절한 AI 도우미다. "
    "보이는 것만 말하고, 확실하지 않은 건 '추정'이라고 표시한다."
)

SYSTEM_CURATOR = (
    "너는 미술관 큐레이터다. 사용자가 올린 이미지를 작품 감상하듯 전문적으로 설명한다.\n"
    "규칙:\n"
    "1) 보이는 것 기반으로 말한다. 작품명/작가/연도는 확실하지 않으면 '추정'으로만 말한다.\n"
    "2) 근거 없이 단정하지 않는다. 모르면 모른다고 말한다.\n"
    "3) 출력은 아래 포맷을 반드시 따른다.\n\n"
    "[전시 라벨]\n"
    "- 1~2문장으로 핵심 소개\n\n"
    "[큐레이터 해설]\n"
    "- 5~8문장: 구도/색/빛/질감/기법/분위기 중심\n\n"
    "[관람 포인트 3]\n"
    "1) ...\n"
    "2) ...\n"
    "3) ...\n\n"
    "[확실한 것 / 추정인 것]\n"
    "- 확실: ...\n"
    "- 추정: ...\n"
)

SYSTEM_SNS = (
    "너는 인스타그램 감상평을 잘 쓰는 작성자다. 한국어로 MZ스럽고 트렌디하게, "
    "짧게(4~7줄) 쓰고 마지막에 해시태그 8~15개를 붙인다.\n"
    "중요 규칙:\n"
    "1) 이미지에서 확실히 알 수 없는 작가명/작품명/전시명/장소/연도는 절대 지어내지 마라.\n"
    "2) 작가/전시를 특정할 수 없으면 해시태그는 #작가미상 #전시정보없음 같은 형태로 처리.\n"
    "3) 문장 사이에 이모지 1~3개만 자연스럽게 사용.\n"
    "4) 해시태그는 두 묶음으로 출력:\n"
    "   - [확실 태그] : 관찰 가능한 요소 기반(색/분위기/주제/소재)\n"
    "   - [추정 태그] : '추정'을 포함한 태그만(예: #추정_인상주의)\n"
    "출력 형식:\n"
    "(감상평 4~7줄)\n"
    "[확실 태그]\n"
    "#... #... #...\n"
    "[추정 태그]\n"
    "#추정_... #추정_...\n"
)

# 1차 관찰 메모(2단계 리라이트용): 환각 줄이는 안전장치
SYSTEM_OBSERVE = (
    "너는 매우 신중한 시각 분석가다. 이미지에서 '보이는 사실'만 추출한다.\n"
    "반드시 아래 JSON 비슷한 형태로만 작성:\n"
    "FACTS: (관찰 가능한 사실 bullet)\n"
    "STYLE_GUESSES: (가능한 사조/스타일 추정 bullet, 반드시 '추정' 표기)\n"
    "UNSURE: (확신 못하는 것 bullet)\n"
)

SYSTEM_REWRITE = (
    "너는 글을 다듬는 편집자다. 아래 '관찰 메모'에 들어있는 내용만 사용해서, "
    "요청한 톤과 포맷으로 최종 답변을 만든다. "
    "관찰 메모에 없는 새로운 사실(작가/작품명/연도 등)은 추가하지 마라."
)

def _data_url_from_upload(uploaded):
    img_bytes = uploaded.getvalue()
    mime = uploaded.type or "application/octet-stream"
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}", img_bytes, mime

def _cache_key(img_bytes: bytes, prompt: str, mode: str, two_pass: bool):
    h = hashlib.sha256()
    h.update(img_bytes)
    h.update(prompt.encode("utf-8"))
    h.update(mode.encode("utf-8"))
    h.update(str(two_pass).encode("utf-8"))
    return h.hexdigest()

def _call_chat(messages, max_tokens=800, temperature=0.7, top_p=0.95):
    completion = client.chat.completions.create(
        model=azure_oai_deployment,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return completion.choices[0].message.content

# -------------------------
# UI (미니지만 제품처럼 보이게)
# -------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    mode = st.selectbox("모드", ["일반 설명", "큐레이터 해설", "SNS 감상문"], index=1)
    two_pass = st.checkbox("2단계 리라이트(더 그럴듯하게)", value=True)
    st.caption(f"모델: `{azure_oai_deployment}`")

uploaded = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg", "webp"])
prompt = st.text_input("질문", value="이 이미지(또는 작품)를 큐레이터처럼 설명해줘.")

if uploaded is not None:
    st.image(uploaded, caption="업로드된 이미지", use_container_width=True)

# 간단 캐시(세션)
if "result_cache" not in st.session_state:
    st.session_state["result_cache"] = {}

if st.button("전송") and uploaded is not None:
    data_url, img_bytes, mime = _data_url_from_upload(uploaded)
    ck = _cache_key(img_bytes, prompt, mode, two_pass)

    if ck in st.session_state["result_cache"]:
        st.subheader("결과(캐시)")
        st.write(st.session_state["result_cache"][ck])
        st.stop()

    with st.spinner("해설 만드는 중... 큐레이터가 전시실 뛰어오는 중입니다 🏃‍♂️"):
        try:
            # 모드별 시스템 프롬프트
            if mode == "일반 설명":
                system = SYSTEM_GENERAL
                max_tokens = 700
                temperature = 0.5
            elif mode == "큐레이터 해설":
                system = SYSTEM_CURATOR
                max_tokens = 900
                temperature = 0.6
            else:
                system = SYSTEM_SNS
                max_tokens = 250
                temperature = 0.8

            if not two_pass:
                # 1-pass: 바로 답변
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]},
                ]
                out = _call_chat(messages, max_tokens=max_tokens, temperature=temperature)

            else:
                # 2-pass: (1) 관찰 메모 → (2) 최종 작성
                observe_messages = [
                    {"role": "system", "content": SYSTEM_OBSERVE},
                    {"role": "user", "content": [
                        {"type": "text", "text": "이미지에서 보이는 사실만 추출해줘."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ]},
                ]
                memo = _call_chat(observe_messages, max_tokens=600, temperature=0.2)

                final_messages = [
                    {"role": "system", "content": SYSTEM_REWRITE},
                    {"role": "user", "content": (
                        f"요청 모드: {mode}\n\n"
                        f"원래 질문: {prompt}\n\n"
                        f"관찰 메모:\n{memo}\n\n"
                        f"이 관찰 메모에 있는 내용만 사용해서 최종 답변을 작성해줘.\n"
                        f"큐레이터 모드면 지정 포맷을 지켜줘."
                    )},
                    {"role": "system", "content": system},
                ]
                out = _call_chat(final_messages, max_tokens=max_tokens, temperature=temperature)

                with st.expander("🔎 1차 관찰 메모(숨김)", expanded=False):
                    st.code(memo)

            st.subheader("결과")
            st.write(out)
            st.session_state["result_cache"][ck] = out

        except Exception as e:
            st.error("호출 실패")
            st.exception(e)
else:
    st.caption("이미지 업로드 후 전송을 눌러주세요.")


