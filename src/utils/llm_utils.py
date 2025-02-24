from typing import Dict, Any, List
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from src.config.config import OLLAMA_BASE_URL, TOP_K_RESULTS
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()


class LLMManager:
    """Simplified LLM manager using two models for extraction and reasoning."""

    def __init__(self):
        # Model for extracting relevant content
        self.extractor_model = Ollama(
            base_url=OLLAMA_BASE_URL,
            model="llama3.2:latest",
            temperature=0.1,  # Low temperature for focused extraction
            num_ctx=2048
        )

        # Model for generating answers
        self.reasoning_model = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.thread_pool = ThreadPoolExecutor(max_workers=2)

    def create_chain(self, vectorstore: Any, model_name: str = "gpt-3.5-turbo") -> Any:
        """Create a simple retriever with configurable settings.
        
        Args:
            vectorstore: The vector store to create a retriever from
            model_name: Name of the model to use (default: gpt-3.5-turbo)
            
        Returns:
            A configured retriever
        """
        # Update the reasoning model based on selected model
        if model_name != self.reasoning_model.model_name:
            self.reasoning_model = ChatOpenAI(
                model_name=model_name,
                temperature=0.7,
                max_tokens=1000,
                api_key=os.getenv("OPENAI_API_KEY")
            )

        # Create retriever for single document
        return vectorstore.as_retriever(
            search_kwargs={"k": 1}  # Get only 1 most relevant document
        )

    async def process_query(self, chain, query: str) -> Dict[str, Any]:
        """Process query using a simplified single-document approach."""
        try:
            # Step 1: Get the most relevant document
            try:
                relevant_docs = chain.get_relevant_documents(query)
            except Exception as e:
                print(f"Document retrieval error: {str(e)}")
                return {
                    "answer": "Error retrieving relevant document. Please try again.",
                    "extracted_content": [],
                    "source_count": 0
                }

            if not relevant_docs:
                return {
                    "answer": "No relevant information found. Please try a different question.",
                    "extracted_content": [],
                    "source_count": 0
                }

            # Step 2: Extract content from the single most relevant document
            doc = relevant_docs[0]  # Get first document
            extraction_prompt = f"""
            Query: {query}

            Document:
            {doc.page_content}

            Extract only the most relevant information to answer the query. 
            Keep the response focused and under 200 words.
            If no relevant information exists, respond with 'NO_RELEVANT_CONTENT'.
            """

            try:
                extracted_content = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    lambda: self.extractor_model.invoke(extraction_prompt)
                )
                
                if not extracted_content or extracted_content.strip().lower().startswith("no_relevant_content"):
                    return {
                        "answer": "Couldn't find relevant information to answer your query.",
                        "extracted_content": [],
                        "source_count": 0
                    }
                
                extracted_content = extracted_content.strip()
                
            except Exception as e:
                print(f"Extraction error: {str(e)}")
                return {
                    "answer": "Error extracting information. Please try again.",
                    "extracted_content": [],
                    "source_count": 0
                }

            # Step 3: Generate answer
            reasoning_prompt = f"""
            Query: {query}

            Relevant Information:
            {extracted_content}

            Provide a clear and direct answer based on the information above.
            If you can't fully answer the query with the given information, say so.
            """

            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    lambda: self.reasoning_model.invoke(reasoning_prompt)
                )
                response_content = response.content if hasattr(
                    response, 'content') else str(response)
            except Exception as e:
                print(f"Reasoning error: {str(e)}")
                return {
                    "answer": "Error generating response. Please try again.",
                    "extracted_content": [{"content": extracted_content}],
                    "source_count": 1
                }

            return {
                "answer": response_content,
                "extracted_content": [{"content": extracted_content}],
                "source_count": 1
            }

        except Exception as e:
            print(f"Processing error: {str(e)}")
            return {
                "answer": "An error occurred. Please try again.",
                "extracted_content": [],
                "source_count": 0
            }
