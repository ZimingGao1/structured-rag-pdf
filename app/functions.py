
import asyncio
import logging
import os
import re
import tempfile
import uuid
import pandas as pd
from functools import lru_cache
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_filename(filename):
    """Remove '(number)' pattern from filename to avoid Chroma collection name errors."""
    if not isinstance(filename, str):
        raise ValueError("Filename must be a string")
    return re.sub(r'\s\(\d+\)', '', filename)

def get_pdf_text(uploaded_file):
    """Load a PDF document from an uploaded file and return it as a list of documents."""
    if not uploaded_file:
        raise ValueError("No file provided")
    
    try:
        input_file = uploaded_file.read()
        if not input_file:
            raise ValueError("Uploaded file is empty")
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_file.write(input_file)
        temp_file.close()

        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages from PDF")
        
        return documents
    
    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}")
        raise
    
    finally:
        try:
            os.unlink(temp_file.name)
        except Exception as e:
            logger.warning(f"Failed to delete temp file: {str(e)}")

def split_document(documents, chunk_size=600, chunk_overlap=100):
    """Split documents into smaller chunks."""
    if not documents:
        raise ValueError("No documents provided for splitting")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " "],
        add_start_index=True
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks")
    return chunks

@lru_cache(maxsize=1)
def get_embedding_function(api_key):
    """Return a cached OpenAIEmbeddings object."""
    if not api_key:
        raise ValueError("API key is required")
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=api_key,
        openai_api_base="https://chatapi.littlewheat.com/v1",
    )
    logger.info("Initialized embedding function")
    return embeddings

def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db"):
    """Create a vector store from a list of text chunks."""
    if not chunks:
        raise ValueError("No chunks provided for vector store creation")
    
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
    unique_ids = set()
    unique_chunks = []
    
    for chunk, id in zip(chunks, ids):
        if id not in unique_ids:
            unique_ids.add(id)
            unique_chunks.append(chunk)
    
    if not unique_chunks:
        raise ValueError("No unique chunks after deduplication")
    
    collection_name = clean_filename(file_name) + f"_{uuid.uuid4().hex[:8]}"
    vectorstore = Chroma.from_documents(
        documents=unique_chunks,
        collection_name=collection_name,
        embedding=embedding_function,
        ids=list(unique_ids),
        persist_directory=vector_store_path
    )
    
    vectorstore.persist()
    logger.info(f"Created vector store with {len(unique_chunks)} documents")
    return vectorstore

def create_vectorstore_from_texts(documents, api_key, file_name):
    """Create a vector store from a list of texts."""
    try:
        docs = split_document(documents)
        embedding_function = get_embedding_function(api_key)
        vectorstore = create_vectorstore(docs, embedding_function, file_name)
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise

def load_vectorstore(file_name, api_key, vectorstore_path="db"):
    """Load a previously saved Chroma vector store from disk."""
    try:
        embedding_function = get_embedding_function(api_key)
        vectorstore = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=embedding_function,
            collection_name=clean_filename(file_name)
        )
        logger.info(f"Loaded vector store for {file_name}")
        return vectorstore
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        raise

# Enhanced Prompt Template
PROMPT_TEMPLATE = """
你是一个从表格中提取信息的专家助手。请根据提供的上下文准确提取以下字段，上下文可能包含表格行和列结构（如第 1 行文献特性）。提取字段包括：文献编号、文献特性、作者信息、干预措施、适应症、研究目标、研究方法、临床结果（可能有多个，如 Clinical Outcome 1, 2, 3 等）、性能/安全性、其他信息。如果某个字段在上下文中不存在，请回答“未在提供的上下文中找到”。不要虚构或假设任何未明确出现的信息。

上下文：
{context}

---

问题：从上下文中提取文献编号、文献特性、作者信息、干预措施、适应症、研究目标、研究方法、临床结果（可能有多个）、性能/安全性、其他信息。
"""

class AnswerWithSources(BaseModel):
    """An answer to the question, with sources and reasoning."""
    answer: str = Field(description="Answer to question")
    sources: str = Field(description="Full direct text chunk from the context used to answer the question")
    reasoning: str = Field(description="Explain the reasoning of the answer based on the sources")

class ClinicalOutcome(BaseModel):
    """Clinical outcome field with number and content."""
    outcome_number: str = Field(description="Clinical outcome number, e.g., Clinical Outcome 1")
    outcome_content: AnswerWithSources = Field(description="Clinical outcome content")

class ExtractedInfoWithSources(BaseModel):
    """Extracted table information."""
    literature_number: AnswerWithSources = Field(description="Literature number")
    literature: AnswerWithSources = Field(description="Literature characteristics")
    author: AnswerWithSources = Field(description="Author information")
    device_intervention: AnswerWithSources = Field(description="Device/Intervention")
    indication: AnswerWithSources = Field(description="Indication")
    study_objective: AnswerWithSources = Field(description="Study Objective")
    method: AnswerWithSources = Field(description="Method")
    clinical_outcomes: List[ClinicalOutcome] = Field(description="List of clinical outcomes")
    performance_safety: AnswerWithSources = Field(description="Performance/Safety")
    other: AnswerWithSources = Field(description="Other information")

def format_docs(docs):
    """Format a list of Document objects into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

async def query_document(vectorstore, query, api_key):
    """Query a vector store with a question and return a structured response."""
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_base="https://chatapi.littlewheat.com/v1",
            openai_api_key=api_key,
        )
        
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )
        
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm.with_structured_output(ExtractedInfoWithSources, strict=True)
        )
        
        structured_response = await rag_chain.ainvoke(query)
        response_dict = structured_response.dict()

        # Build result DataFrame
        clinical_outcomes = response_dict["clinical_outcomes"]
        clinical_outcome_columns = {}
        for outcome in clinical_outcomes:
            outcome_num = outcome["outcome_number"]
            clinical_outcome_columns[f"{outcome_num}"] = [outcome["outcome_content"]["answer"]]
            clinical_outcome_columns[f"{outcome_num}_来源"] = [outcome["outcome_content"]["sources"]]
            clinical_outcome_columns[f"{outcome_num}_推理"] = [outcome["outcome_content"]["reasoning"]]

        result_df = pd.DataFrame({
            "文献编号": [response_dict["literature_number"]["answer"]],
            "文献编号_来源": [response_dict["literature_number"]["sources"]],
            "文献编号_推理": [response_dict["literature_number"]["reasoning"]],
            "文献特性": [response_dict["literature"]["answer"]],
            "文献特性_来源": [response_dict["literature"]["sources"]],
            "文献特性_推理": [response_dict["literature"]["reasoning"]],
            "作者信息": [response_dict["author"]["answer"]],
            "作者信息_来源": [response_dict["author"]["sources"]],
            "作者信息_推理": [response_dict["author"]["reasoning"]],
            "干预措施": [response_dict["device_intervention"]["answer"]],
            "干预措施_来源": [response_dict["device_intervention"]["sources"]],
            "干预措施_推理": [response_dict["device_intervention"]["reasoning"]],
            "适应症": [response_dict["indication"]["answer"]],
            "适应症_来源": [response_dict["indication"]["sources"]],
            "适应症_推理": [response_dict["indication"]["reasoning"]],
            "研究目标": [response_dict["study_objective"]["answer"]],
            "研究目标_来源": [response_dict["study_objective"]["sources"]],
            "研究目标_推理": [response_dict["study_objective"]["reasoning"]],
            "研究方法": [response_dict["method"]["answer"]],
            "研究方法_来源": [response_dict["method"]["sources"]],
            "研究方法_推理": [response_dict["method"]["reasoning"]],
            **clinical_outcome_columns,
            "性能/安全性": [response_dict["performance_safety"]["answer"]],
            "性能/安全性_来源": [response_dict["performance_safety"]["sources"]],
            "性能/安全性_推理": [response_dict["performance_safety"]["reasoning"]],
            "其他信息": [response_dict["other"]["answer"]],
            "其他信息_来源": [response_dict["other"]["sources"]],
            "其他信息_推理": [response_dict["other"]["reasoning"]],
        })
        
        logger.info("Successfully queried vector store")
        return result_df
    
    except Exception as e:
        logger.error(f"Error querying vector store: {str(e)}")
        raise