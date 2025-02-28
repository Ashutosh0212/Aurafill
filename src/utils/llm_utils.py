from typing import Dict, Any, List
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from src.config.config import OLLAMA_BASE_URL, TOP_K_RESULTS, CHUNK_SIZE, CHUNK_OVERLAP
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

load_dotenv()


class LLMManager:
    """Simplified LLM manager using two models for extraction and reasoning."""

    def __init__(self):
        # Model for extracting relevant content
        self.extractor_model = Ollama(
            base_url=OLLAMA_BASE_URL,
            model="llama3.2:latest",
            temperature=0.1,  # Low temperature for focused extraction
            num_ctx=4096
        )

        # Model for generating answers
        self.reasoning_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=2024,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.thread_pool = ThreadPoolExecutor(max_workers=2)

        # Improved text splitter with better defaults
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )

    def create_chain(self, vectorstore: Chroma, model_name: str = "gpt-3.5-turbo") -> Any:
        """Create an advanced retriever with better search capabilities."""
        # Update the reasoning model
        if model_name != self.reasoning_model.model_name:
            self.reasoning_model = ChatOpenAI(
                model_name=model_name,
                temperature=0.7,
                max_tokens=2024,
                api_key=os.getenv("OPENAI_API_KEY")
            )

        # Create base retriever with similarity search
        base_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": TOP_K_RESULTS,
            }
        )

        # Create an LLM-based compressor for better extraction
        compressor = LLMChainExtractor.from_llm(self.extractor_model)

        # Create a contextual compression retriever
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )

    def process_documents(self, text_content: str) -> List[LangchainDocument]:
        """Split documents into chunks for better retrieval."""
        # Split text into chunks
        chunks = self.text_splitter.split_text(text_content)
        
        # Convert chunks to Langchain documents
        documents = [
            LangchainDocument(
                page_content=chunk,
                metadata={"chunk_id": i}
            ) for i, chunk in enumerate(chunks)
        ]
        
        return documents

    async def process_query(self, chain: ContextualCompressionRetriever, query: str) -> Dict[str, Any]:
        """Process query with improved document retrieval and extraction."""
        try:
            # Get relevant documents with similarity scores and compression
            relevant_docs = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: chain.get_relevant_documents(query)
            )

            if not relevant_docs:
                return {
                    "answer": "No relevant information found. Please try a different question.",
                    "extracted_content": [],
                    "source_count": 0
                }

            # Process each document with improved context
            all_extracted_content = []
            for doc in relevant_docs:
                extraction_prompt = f"""
                Query: {query}

                Document Content:
                {doc.page_content}

                Instructions:
                1. Analyze the content thoroughly for ALL information relevant to the query
                2. Extract both direct answers and supporting context
                3. Include technical details and examples if present
                4. Maintain the original technical accuracy
                5. Format the extraction clearly with:
                   - Main points
                   - Supporting details
                   - Technical specifications (if any)
                   - Examples or use cases (if any)
                6. If no relevant information exists, respond with 'NO_RELEVANT_CONTENT'

                Keep the response focused and well-structured.
                """

                try:
                    extracted_content = await asyncio.get_event_loop().run_in_executor(
                        self.thread_pool,
                        lambda: self.extractor_model.invoke(extraction_prompt)
                    )
                    
                    if extracted_content and not extracted_content.strip().lower().startswith("no_relevant_content"):
                        all_extracted_content.append({
                            "content": extracted_content.strip(),
                            "metadata": doc.metadata if hasattr(doc, 'metadata') else {}
                        })
                        
                except Exception as e:
                    print(f"Extraction error for document: {str(e)}")
                    continue

            if not all_extracted_content:
                return {
                    "answer": "Couldn't find relevant information to answer your query.",
                    "extracted_content": [],
                    "source_count": 0
                }

            # Combine extracted content with better structure
            combined_content = "\n\n".join(
                f"Source {i+1}:\n{content['content']}"
                for i, content in enumerate(all_extracted_content)
            )

            # Generate comprehensive answer
            reasoning_prompt = f"""
            Query: {query}

            Information from {len(all_extracted_content)} relevant sources:
            {combined_content}

            Instructions:
            1. this is for a Virtualization and Cloud Computing course held by Dr. Sumit karla
            2. TA's in this course are Manik Sejwal, Arpit, Satish Ray, Sangeeta
            1. Remember that you are a expert in the field of VCC and you are helping the user to get the best answer for their query.
            2. You are given a list of sources that contain information about the query.
            3. You need to synthesize a comprehensive answer using ALL provided sources.
            2. Ensure the answer is accurate and relevant to the query
            3. If the answer is not found in the sources, respond with "No relevant information found."
            4. Ensure the answer is well-structured and easy to understand
            5. please dont use the word "sources in the answer
            5. Aim for a complete, well-organized response that fully addresses the query.
            """

            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    lambda: self.reasoning_model.invoke(reasoning_prompt)
                )
                response_content = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                print(f"Reasoning error: {str(e)}")
                return {
                    "answer": "Error generating response. Please try again.",
                    "extracted_content": all_extracted_content,
                    "source_count": len(relevant_docs)
                }

            return {
                "answer": response_content,
                "extracted_content": all_extracted_content,
                "source_count": len(relevant_docs)
            }

        except Exception as e:
            print(f"Processing error: {str(e)}")
            return {
                "answer": "An error occurred. Please try again.",
                "extracted_content": [],
                "source_count": 0
            }