import streamlit as st
import uuid
from Agent import load_agent, KB_TOPICS

# ── Page config ──────────────────────────────────────────
st.set_page_config(page_title="AI Course Assistant", layout="centered")
st.title("🤖 AI Course Assistant")
st.caption("An AI that helps with Agentic AI course related questions.")

DOMAIN_DESCRIPTION = "An AI that helps with Agentic AI course related questions."

# ── Load agent (cached) ──────────────────────────────────
@st.cache_resource
def get_agent():
    return load_agent()

try:
    agent_app, embedder, collection = get_agent()
    st.success(f"✅ Knowledge base loaded — {collection.count()} documents")
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.stop()

# ── Session state ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.write(DOMAIN_DESCRIPTION)
    st.write(f"Session: {st.session_state.thread_id}")
    st.divider()
    st.write("**Topics covered:**")
    for t in KB_TOPICS:
        st.write(f"• {t}")
    if st.button("🗑️ New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

# ── Display history ──────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Chat input ───────────────────────────────────────────
if prompt := st.chat_input("Ask something..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent_app.invoke(
                {"question": prompt,
                 "messages": st.session_state.messages},
                config={"configurable": {"thread_id": st.session_state.thread_id}}
            )
            answer = result.get("answer", "Sorry, I could not generate an answer.")
        st.write(answer)
        faith = result.get("faithfulness", 0.0)
        if faith > 0:
            st.caption(f"Faithfulness: {faith:.2f} | Sources: {result.get('sources', [])}")

    st.session_state.messages.append({"role": "assistant", "content": answer})