import json
import os
import re

from dotenv import load_dotenv
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_openai import ChatOpenAI
from underthesea import ner

# Load biến môi trường từ file .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Khởi tạo mô hình LLM với cấu hình
llm = ChatOpenAI(
    temperature=0,
    streaming=True,
    model="gpt-4o-mini",
    openai_api_key=openai_api_key,
)

# Định nghĩa mẫu prompt
system = """You are an expert at Stock. Your name is StockAI.
Here is the context you should refer to:
{context}
Your response must be in Vietnamese and must be in json format
    question: User's question
    answer: Your answer
If there is no relevant documentation, reply no data.
"""

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "Question: {question}")]
)

# Khởi tạo chuỗi RAG với prompt_template
rag_chain = (
    RunnableMap(
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    )
    | prompt_template
    | llm
    | StrOutputParser()
)


# Hàm trích xuất ngày tháng từ văn bản với regex
def extract_date_with_regex(query):
    date_pattern = r"\b(\d{1,2}/\d{1,2}/\d{4})\b"
    match = re.search(date_pattern, query)
    if match:
        return match.group(0)
    return None


# Hàm trích xuất ngày tháng với NER
def extract_date_with_ner(query):
    entities = ner(query)
    for entity in entities:
        if entity[3] == "B-DATE":
            return entity[0]
    return None


# Hàm kết hợp hai phương pháp trích xuất ngày tháng
def extract_date_from_query(query):
    date_text = extract_date_with_regex(query)
    if date_text:
        return date_text
    return extract_date_with_ner(query)


def remove_json_formatting(input_text):
    # Loại bỏ dấu ```json và ``` nếu chúng có trong input_text
    cleaned_text = input_text.strip("```json").strip("```").strip()
    return cleaned_text


# Hàm tạo retriever từ Chroma và BM25
def create_retriever(vector_db, query, k=4):
    # Tạo điều kiện lọc dựa vào ngày tháng
    date_text = extract_date_from_query(query)
    filter_conditions = {"date": date_text} if date_text else None

    # Tạo chroma retriever với điều kiện lọc
    chroma_retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs=(
            {"k": k, "filter": filter_conditions}
            if filter_conditions
            else {"k": k}
        ),
    )

    # Tạo BM25 retriever
    documents_bm25 = [
        Document(page_content=doc.page_content, metadata=doc.metadata)
        for doc in vector_db.similarity_search("", k=100)
    ]
    bm25_retriever = BM25Retriever.from_documents(documents_bm25)
    bm25_retriever.k = k

    # Kết hợp retriever (chroma + BM25) thành ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever], weights=[0.7, 0.3]
    )
    return ensemble_retriever, filter_conditions


# Hàm lấy tài liệu phù hợp từ retriever
def retrieve_documents(ensemble_retriever, query, top_k=4):
    docs = ensemble_retriever.get_relevant_documents(query=query)
    return docs[:top_k]


# Hàm định dạng tài liệu thành chuỗi
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


# Hàm gọi chuỗi RAG và sinh câu trả lời
def generate_answer(vector_db, query, top_k=4):
    ensemble_retriever, filter_conditions = create_retriever(
        vector_db, query, k=top_k
    )
    docs = retrieve_documents(ensemble_retriever, query, top_k=top_k)
    formatted_docs = format_docs(docs)
    output = rag_chain.invoke({"context": formatted_docs, "question": query})
    output = remove_json_formatting(output)
    return output
