import os
import base64
import hashlib
import streamlit as st
from openai import AzureOpenAI

# (선택) 로컬 개발용 .env 지원: 배포에서 없어도 안 죽게
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

st.set_page_config(page_title="이미지 설명 챗봇", page_icon="🖼️", layout="centered")
st.title("🖼️ 이미지 설명 챗봇 (Azure OpenAI)")

# -------------------------
# 환경변수 / Secrets
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
# 프롬프트
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

# 인스타 감상문(MZ + 해시태그, 단 작가/전시/작품명은 모르면 안 지어냄)
SYSTEM_SNS = (
    "너는 인스타그램 감상평을 잘 쓰는 작성자다. 한국어로 트렌디하게, "
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

# -------------------------
# 유틸
# -------------------------
APP_CACHE_VERSION = "v1"  # 코드 바꿨을 때 캐시 키 무효화용(원하면 값 바꾸세요)

def data_url_from_upload(uploaded):
    img_bytes = uploaded.getvalue()
    mime = uploaded.type or "application/octet-stream"
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}", img_bytes

def cache_key(img_bytes: bytes, prompt: str, mode: str):
    h = hashlib.sha256()
    h.update(APP_CACHE_VERSION.encode("utf-8"))
    h.update(img_bytes)
    h.update(prompt.encode("utf-8"))
    h.update(mode.encode("utf-8"))
    return h.hexdigest()

def call_chat(messages, max_tokens=800, temperature=0.7, top_p=0.95):
    completion = client.chat.completions.create(
        model=azure_oai_deployment,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return completion.choices[0].message.content

# -------------------------
# 상태(캐시)
# -------------------------
if "result_cache" not in st.session_state:
    st.session_state["result_cache"] = {}

# -------------------------
# UI
# -------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    mode = st.selectbox("모드", ["일반 설명", "큐레이터 해설", "SNS 감상문"], index=1)
    st.caption(f"모델: `{azure_oai_deployment}`")
    if st.button("🧹 캐시 비우기"):
        st.session_state["result_cache"].clear()
        st.success("캐시를 비웠습니다.")

uploaded = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg", "webp"])
prompt_default = "이 사진 설명해봐." if mode != "큐레이터 해설" else "이 이미지(또는 작품)를 큐레이터처럼 설명해줘."
prompt = st.text_input("질문", value=prompt_default)

if uploaded is not None:
    st.image(uploaded, caption="업로드된 이미지", use_container_width=True)

col1, col2 = st.columns(2)
send_clicked = col1.button("전송")
regen_clicked = col2.button("답변 다시 생성 🔄", help="캐시를 무시하고 모델을 다시 호출합니다.")

# -------------------------
# 실행
# -------------------------
if (send_clicked or regen_clicked) and uploaded is not None:
    force_regen = regen_clicked  # ✅ 다시 생성이면 캐시 무시

    data_url, img_bytes = data_url_from_upload(uploaded)
    ck = cache_key(img_bytes, prompt, mode)

    # ✅ 캐시 재사용(단, 다시 생성이면 무시)
    if (not force_regen) and (ck in st.session_state["result_cache"]):
        st.subheader("결과(캐시)")
        st.write(st.session_state["result_cache"][ck])
        st.stop()

    # 모드별 프롬프트/기본 파라미터
    if mode == "일반 설명":
        system = SYSTEM_GENERAL
        base_temp = 0.5
        max_tokens = 700
    elif mode == "큐레이터 해설":
        system = SYSTEM_CURATOR
        base_temp = 0.6
        max_tokens = 900
    else:
        system = SYSTEM_SNS
        base_temp = 0.8
        max_tokens = 300

    # ✅ 다시 생성일 때만 temperature 살짝 올리기(변주 강화)
    temperature = min(base_temp + (0.2 if force_regen else 0.0), 1.2)

    with st.spinner("생성 중... 모델도 사람처럼 컨디션이 있습니다(농담)."):
        try:
            messages = [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ]

            out = call_chat(messages, max_tokens=max_tokens, temperature=temperature)

            st.subheader("결과(새로 생성)" if force_regen else "결과")
            st.write(out)

            # ✅ 새 결과로 캐시 갱신(덮어쓰기)
            st.session_state["result_cache"][ck] = out

        except Exception as e:
            st.error("호출 실패")
            st.exception(e)
else:
    st.caption("이미지 업로드 후 전송 또는 ‘답변 다시 생성’을 눌러주세요.")



