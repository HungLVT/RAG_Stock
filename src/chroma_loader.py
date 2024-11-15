# from langchain.schema import Document
import csv

# import pandas as pd
# from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# def load_data(file_path):
#     """
#     Load data from a CSV file and return as a list of Document objects.
#     """
#     file_path = "../data/news_data.csv"
#     df = pd.read_csv(file_path, encoding="utf-8")
#     temp_file_path = "temp_news_data_utf8.csv"
#     df.to_csv(temp_file_path, index=False, encoding="utf-8")

#     loader = CSVLoader(temp_file_path)
#     documents = loader.load()
#     updated_documents = []

#     for doc in documents:
#         # Tách các phần trong page_content bằng ký tự xuống dòng '\n'
#         parts = doc.page_content.split("\n")

#         # Kiểm tra nếu có ít nhất 4 dòng để đảm bảo dòng 'date' tồn tại
#         if len(parts) >= 4:
#             date_line = parts[
#                 3
#             ].strip()  # Lấy dòng thứ 4 và loại bỏ khoảng trắng
#             date = date_line.replace("date: ", "").split()[
#                 0
#             ]  # Loại bỏ tiền tố 'date: ' và lấy phần ngày

#             # Thêm 'date' vào metadata
#             doc.metadata["date"] = date

#         updated_documents.append(doc)
#     return updated_documents


def load_data(file_path):
    csv.field_size_limit(10**6)

    loader = CSVLoader(file_path, encoding="utf-8")

    documents = loader.load()
    updated_documents = []

    for doc in documents:
        # Tách các phần trong page_content bằng ký tự xuống dòng '\n'
        parts = doc.page_content.split("\n")

        # Kiểm tra nếu có ít nhất 4 dòng để đảm bảo dòng 'date' tồn tại
        if len(parts) >= 4:
            date_line = parts[
                3
            ].strip()  # Lấy dòng thứ 4 và loại bỏ khoảng trắng
            date = date_line.replace("date: ", "").split()[
                0
            ]  # Loại bỏ tiền tố 'date: ' và lấy phần ngày

            # Thêm 'date' vào metadata
            doc.metadata["date"] = date

        updated_documents.append(doc)
    return updated_documents[:100]


# def load_data(file_path):
#     """
#     Load data from a CSV file and return as a list of Document objects.
#     """
#     try:
#         # Đọc file CSV với pandas và xử lý NaN bằng chuỗi rỗng
#         df = pd.read_csv(file_path, encoding="utf-8")
#         df["content"] = df["content"].fillna("")  # Thay NaN bằng chuỗi rỗng

#         # Tạo danh sách Document từ mỗi dòng của DataFrame
#         documents = [
#             Document(page_content=row["content"], metadata=row.to_dict())
#             for _, row in df.iterrows()
#         ]

#         updated_documents = []
#         for doc in documents:
#             parts = doc.page_content.split("\n")

#             if len(parts) >= 4:
#                 date_line = parts[3].strip()
#                 date = date_line.replace("date: ", "").split()[0]
#                 doc.metadata["date"] = date

#             updated_documents.append(doc)
#         return updated_documents
#     except UnicodeDecodeError as e:
#         print(f"Encoding error: {e}. Please check the file encoding.")
#         return []
#     except Exception as e:
#         print(f"Error loading data: {e}")
#         return []


def create_chroma_db(documents, persist_directory="vector_db"):
    print("Creating Chroma database...")

    # Tách văn bản thành các đoạn nhỏ hơn
    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(
        "Documents split successfully. Number of documents after splitting:",
        len(docs),
    )

    # Khởi tạo embeddings, đảm bảo mô hình chạy trên CPU để tránh lỗi GPU
    embeddings = HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
    )
    print("Embeddings loaded successfully.")

    # Tạo Chroma database từ các documents đã chia nhỏ, với embedding và
    # lưu vào `persist_directory`
    chroma_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    print("Chroma database created successfully.")

    # Xác nhận việc lưu Chroma database vào thư mục đã chỉ định
    print(f"Chroma database saved to {persist_directory}")
    return chroma_db


def load_existing_chroma_db(persist_directory="vector_db"):
    # Tạo hàm embedding giống như khi bạn tạo vector_db ban đầu
    embeddings = HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ),
    )

    # Tải cơ sở dữ liệu Chroma đã lưu từ `persist_directory`
    chroma_db = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    print(f"Chroma database loaded from {persist_directory}")
    return chroma_db


# def create_chroma_db(updated_documents, persist_directory="vector_db"):
#     text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
#     docs = text_splitter.split_documents(updated_documents)
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
#     )
#     vector_db = Chroma.from_documents(
#         documents=docs, embedding=embeddings, persist_directory="vector_db"
#     )
#     return vector_db
