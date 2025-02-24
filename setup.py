from setuptools import setup, find_packages

setup(
    name="class_chat_bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit==1.32.0",
        "langchain==0.1.9",
        "chromadb==0.4.22",
        "faiss-cpu==1.7.4",
        "qdrant-client==1.7.0",
        "python-dotenv==1.0.1",
        "langchain-community==0.0.24",
        "langchain-core==0.1.27",
        "pyyaml==6.0.1",
        "python-multipart==0.0.9",
        "typing-extensions==4.9.0"
    ],
) 