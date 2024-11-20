import json

import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)

from chroma_loader import create_chroma_db, load_data
from model_handler import generate_answer

# Đường dẫn tới file CSV của bạn
file_path = "../data/news_data.csv"


def setup_page():
    """
    Cấu hình trang web cơ bản
    """
    st.set_page_config(
        page_title="AI Assistant",
        page_icon="💬",
        layout="wide",
    )


setup_page()


# Sử dụng bộ nhớ đệm để lưu dữ liệu và cơ sở dữ liệu vector
@st.cache_data
def load_documents(file_path):
    return load_data(file_path)


@st.cache_resource
def initialize_vector_db(_documents):
    return create_chroma_db(documents=_documents, persist_directory="vector_db")


def remove_json_formatting(input_text):
    # Loại bỏ dấu ```json và ``` nếu chúng có trong input_text
    cleaned_text = input_text.strip("```json").strip("```").strip()
    return cleaned_text


# Tải dữ liệu và cơ sở dữ liệu vector
documents = load_documents(file_path)
vector_db = initialize_vector_db(documents)

# Thiết lập tiêu đề ứng dụng
st.title("StockAI - Tư vấn Chứng khoán")

# Sử dụng StreamlitChatMessageHistory để quản lý lịch sử trò chuyện
msgs = StreamlitChatMessageHistory(key="chat_messages")

# Khởi tạo session_state cho lịch sử chat nếu chưa có
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Tôi có thể giúp gì cho bạn?"}
    ]
    msgs.add_ai_message("Tôi có thể giúp gì cho bạn?")


# Tải file CSS với encoding UTF-8
with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


with st.container(height=500):
    # st.markdown("<div class='chat-history'>", unsafe_allow_html=True)
    # Hiển thị lịch sử trò chuyện
    for msg in st.session_state.messages:
        role = "assistant" if msg["role"] == "assistant" else "human"
        st.chat_message(role).write(msg["content"])
    # st.markdown("</div>", unsafe_allow_html=True)


# Hàm xử lý câu hỏi và lưu vào lịch sử
def process_question(prompt):
    # Hiển thị tin nhắn của người dùng
    # st.chat_message("human").write(prompt)
    msgs.add_user_message(prompt)
    st.session_state.messages.append({"role": "human", "content": prompt})
    #refresh page

    # Gọi AI để xử lý tin nhắn và nhận phản hồi
    # with st.chat_message("assistant"):
    st_callback = StreamlitCallbackHandler(st.container())

        # Chuẩn bị lịch sử trò chuyện để gửi cho AI
    chat_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[:-1]
    ]

        # Gọi hàm generate_answer
    response = generate_answer(vector_db, prompt)
    response = remove_json_formatting(response)

        # Lưu phản hồi của AI vào lịch sử và hiển thị
    print(response)
    json_output = json.loads(response)
    answer_text = json_output["answer"]
    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text}
    )
    msgs.add_ai_message(answer_text)
        # st.write(answer_text)
    st.experimental_rerun()


# Đảm bảo `user_input` được khởi tạo trong session_state
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Hàm để reset giá trị input
def reset_user_input():
    st.session_state.user_input = ""

# Phần nhập câu hỏi
with st.container():
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    prompt = st.text_input(
        "Nhập câu hỏi về chứng khoán:",
        value=st.session_state.user_input,
        key="user_input",
        placeholder="Nhập câu hỏi của bạn...",
        on_change=reset_user_input,  # Reset sau khi gửi câu hỏi
    )
    if prompt:
        process_question(prompt)
    st.markdown("</div>", unsafe_allow_html=True)



# Đm th Huy béo như con 