from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import os
import logging
from datetime import datetime
import asyncio
import json

# AI/ML imports
import openai
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Monitoring
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Service - Merchant Intelligence",
    description="Retrieval-Augmented Generation service for merchant risk intelligence",
    version="1.0.0"
)

# Metrics
query_counter = Counter('rag_queries_total', 'Total number of RAG queries')
embedding_duration = Histogram('rag_embedding_duration_seconds', 'Time spent on embeddings')
llm_duration = Histogram('rag_llm_duration_seconds', 'Time spent on LLM calls')

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    merchant_id: Optional[str] = None
    context_type: str = Field(default="general", regex="^(general|risk_analysis|compliance|fraud_detection)$")
    include_sources: bool = True
    max_results: int = Field(default=5, ge=1, le=20)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    conversation_id: Optional[str] = None
    merchant_context: Optional[Dict[str, Any]] = None

class DocumentUploadRequest(BaseModel):
    title: str
    category: str = Field(..., regex="^(compliance|risk_guidelines|fraud_patterns|industry_reports)$")
    tags: List[str] = []

class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    query_id: str
    timestamp: datetime

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    sources: List[Dict[str, Any]]
    timestamp: datetime

class RAGService:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.chat_model = None
        self.conversations = {}
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize RAG components"""
        try:
            # Initialize embeddings
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if openai_api_key:
                self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                self.llm = OpenAI(openai_api_key=openai_api_key, temperature=0.1)
                self.chat_model = ChatOpenAI(
                    openai_api_key=openai_api_key,
                    model_name="gpt-3.5-turbo",
                    temperature=0.1
                )
                logger.info("OpenAI components initialized")
            else:
                # Fallback to open-source models
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info("Using HuggingFace embeddings (OpenAI API key not provided)")
            
            # Initialize vector store
            chroma_url = os.getenv('CHROMA_URL', 'http://localhost:8000')
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory="./chroma_db"
            )
            
            # Load initial knowledge base
            self._load_initial_knowledge()
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            raise
    
    def _load_initial_knowledge(self):
        """Load initial knowledge base with merchant risk intelligence"""
        documents = [
            Document(
                page_content="""
                Merchant Risk Assessment Guidelines:
                
                HIGH RISK INDICATORS:
                - Chargeback rate > 2%
                - High-value transactions (>$1000) with new merchants
                - Transactions in high-risk industries (gambling, adult content, pharmaceuticals)
                - Irregular transaction patterns or sudden volume spikes
                - Multiple failed payment attempts
                - Transactions from high-risk countries
                
                MEDIUM RISK INDICATORS:
                - Chargeback rate between 1-2%
                - New merchant with limited transaction history
                - Seasonal business with irregular patterns
                - High refund rates
                
                LOW RISK INDICATORS:
                - Chargeback rate < 1%
                - Established merchant with consistent patterns
                - Low-value, frequent transactions
                - Strong customer authentication
                """,
                metadata={"category": "risk_guidelines", "source": "internal_guidelines"}
            ),
            Document(
                page_content="""
                Fraud Detection Patterns:
                
                VELOCITY FRAUD:
                - Multiple transactions in quick succession
                - Same card used across multiple merchants rapidly
                - Unusual geographic spread of transactions
                
                ACCOUNT TAKEOVER:
                - Login from unusual locations
                - Changes to account information
                - Large transactions after account changes
                
                FRIENDLY FRAUD:
                - Customer disputes legitimate transactions
                - Pattern of chargebacks without returns
                - Claims of non-receipt for digital goods
                
                SYNTHETIC IDENTITY:
                - New accounts with perfect credit
                - Limited credit history
                - Inconsistent personal information
                """,
                metadata={"category": "fraud_patterns", "source": "fraud_team"}
            ),
            Document(
                page_content="""
                Compliance Requirements:
                
                PCI DSS:
                - Secure storage of cardholder data
                - Regular security assessments
                - Access controls and monitoring
                
                KYC/AML:
                - Customer identification verification
                - Beneficial ownership disclosure
                - Transaction monitoring
                - Suspicious activity reporting
                
                GDPR:
                - Data protection and privacy
                - Consent management
                - Right to be forgotten
                - Data breach notification
                
                SOX:
                - Financial reporting controls
                - Internal controls assessment
                - Audit requirements
                """,
                metadata={"category": "compliance", "source": "compliance_team"}
            )
        ]
        
        try:
            # Add documents to vector store
            self.vectorstore.add_documents(documents)
            logger.info(f"Loaded {len(documents)} initial documents")
        except Exception as e:
            logger.error(f"Failed to load initial knowledge: {e}")
    
    async def query_knowledge_base(self, request: QueryRequest) -> Dict[str, Any]:
        """Query the knowledge base with RAG"""
        try:
            with embedding_duration.time():
                # Retrieve relevant documents
                retriever = self.vectorstore.as_retriever(
                    search_kwargs={"k": request.max_results}
                )
                relevant_docs = retriever.get_relevant_documents(request.query)
            
            if not self.llm:
                # Fallback response without LLM
                return {
                    "answer": self._create_fallback_response(relevant_docs, request.query),
                    "sources": [
                        {
                            "content": doc.page_content[:200] + "...",
                            "metadata": doc.metadata,
                            "relevance_score": 0.8
                        }
                        for doc in relevant_docs
                    ],
                    "confidence": 0.7,
                    "query_id": f"query_{datetime.now().timestamp()}",
                    "timestamp": datetime.now()
                }
            
            with llm_duration.time():
                # Create RAG chain
                prompt_template = self._get_prompt_template(request.context_type)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt_template},
                    return_source_documents=True
                )
                
                # Enhanced query with context
                enhanced_query = self._enhance_query(request)
                
                # Get response
                result = qa_chain({"query": enhanced_query})
            
            query_counter.inc()
            
            return {
                "answer": result["result"],
                "sources": [
                    {
                        "content": doc.page_content[:300] + "...",
                        "metadata": doc.metadata,
                        "relevance_score": 0.9
                    }
                    for doc in result["source_documents"]
                ] if request.include_sources else [],
                "confidence": self._calculate_confidence(result),
                "query_id": f"query_{datetime.now().timestamp()}",
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _get_prompt_template(self, context_type: str) -> PromptTemplate:
        """Get context-specific prompt template"""
        templates = {
            "risk_analysis": """
            You are a merchant risk analysis expert. Use the following context to answer questions about merchant risk assessment.
            
            Context: {context}
            
            Question: {question}
            
            Provide a detailed, professional analysis focusing on:
            1. Risk factors and indicators
            2. Recommended actions
            3. Compliance considerations
            4. Industry best practices
            
            Answer:
            """,
            "compliance": """
            You are a compliance expert specializing in payment industry regulations. Use the following context to answer compliance-related questions.
            
            Context: {context}
            
            Question: {question}
            
            Provide a comprehensive answer covering:
            1. Relevant regulations
            2. Compliance requirements
            3. Implementation guidance
            4. Potential penalties for non-compliance
            
            Answer:
            """,
            "fraud_detection": """
            You are a fraud detection specialist. Use the following context to answer questions about fraud patterns and prevention.
            
            Context: {context}
            
            Question: {question}
            
            Provide expert guidance on:
            1. Fraud indicators and patterns
            2. Detection methods
            3. Prevention strategies
            4. Investigation steps
            
            Answer:
            """,
            "general": """
            You are a merchant intelligence assistant. Use the following context to provide helpful, accurate information.
            
            Context: {context}
            
            Question: {question}
            
            Answer:
            """
        }
        
        return PromptTemplate(
            template=templates.get(context_type, templates["general"]),
            input_variables=["context", "question"]
        )
    
    def _enhance_query(self, request: QueryRequest) -> str:
        """Enhance query with additional context"""
        enhanced = request.query
        
        if request.merchant_id:
            enhanced += f" (for merchant ID: {request.merchant_id})"
        
        if request.context_type != "general":
            enhanced += f" (context: {request.context_type})"
        
        return enhanced
    
    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate confidence score based on result quality"""
        # Simple heuristic - can be improved with more sophisticated methods
        answer_length = len(result["result"])
        source_count = len(result.get("source_documents", []))
        
        confidence = min(0.9, (answer_length / 1000) * 0.5 + (source_count / 5) * 0.5)
        return max(0.1, confidence)
    
    def _create_fallback_response(self, docs: List[Document], query: str) -> str:
        """Create fallback response when LLM is not available"""
        if not docs:
            return "No relevant information found in the knowledge base."
        
        # Combine top documents
        combined_content = "\n\n".join([doc.page_content[:500] for doc in docs[:3]])
        return f"Based on the available information:\n\n{combined_content}"
    
    async def chat_conversation(self, request: ChatRequest) -> Dict[str, Any]:
        """Handle conversational chat with memory"""
        conversation_id = request.conversation_id or f"chat_{datetime.now().timestamp()}"
        
        try:
            # Get or create conversation memory
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
            
            memory = self.conversations[conversation_id]
            
            if not self.chat_model:
                # Fallback without conversational AI
                return {
                    "response": "Conversational AI is not available. Please use the query endpoint for information retrieval.",
                    "conversation_id": conversation_id,
                    "sources": [],
                    "timestamp": datetime.now()
                }
            
            # Create conversational retrieval chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.chat_model,
                retriever=self.vectorstore.as_retriever(),
                memory=memory,
                return_source_documents=True
            )
            
            # Add merchant context if provided
            enhanced_message = request.message
            if request.merchant_context:
                context_str = json.dumps(request.merchant_context, indent=2)
                enhanced_message += f"\n\nMerchant Context:\n{context_str}"
            
            # Get response
            result = qa_chain({"question": enhanced_message})
            
            return {
                "response": result["answer"],
                "conversation_id": conversation_id,
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ],
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def add_document(self, content: str, metadata: Dict[str, Any]):
        """Add new document to knowledge base"""
        try:
            # Split document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            chunks = text_splitter.split_text(content)
            documents = [
                Document(page_content=chunk, metadata=metadata)
                for chunk in chunks
            ]
            
            # Add to vector store
            self.vectorstore.add_documents(documents)
            
            logger.info(f"Added document with {len(chunks)} chunks")
            return {"chunks_added": len(chunks)}
            
        except Exception as e:
            logger.error(f"Document addition error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize RAG service
rag_service = RAGService()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "embeddings": rag_service.embeddings is not None,
            "vectorstore": rag_service.vectorstore is not None,
            "llm": rag_service.llm is not None
        },
        "timestamp": datetime.now()
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/query", response_model=RAGResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base using RAG"""
    result = await rag_service.query_knowledge_base(request)
    return RAGResponse(**result)

@app.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(request: ChatRequest):
    """Chat with the AI assistant"""
    result = await rag_service.chat_conversation(request)
    return ChatResponse(**result)

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    title: str = "",
    category: str = "general",
    tags: str = "[]"
):
    """Upload and index a document"""
    try:
        # Read file content
        content = await file.read()
        
        # Handle different file types
        if file.filename.endswith('.pdf'):
            # For PDF files, you would use PyPDFLoader
            # This is a simplified version
            text_content = content.decode('utf-8', errors='ignore')
        else:
            text_content = content.decode('utf-8')
        
        # Parse tags
        try:
            parsed_tags = json.loads(tags)
        except:
            parsed_tags = []
        
        # Add metadata
        metadata = {
            "title": title or file.filename,
            "category": category,
            "tags": parsed_tags,
            "filename": file.filename,
            "upload_date": datetime.now().isoformat(),
            "source": "uploaded"
        }
        
        # Add to knowledge base
        result = await rag_service.add_document(text_content, metadata)
        
        return {
            "message": "Document uploaded successfully",
            "filename": file.filename,
            "chunks_added": result["chunks_added"],
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}/history")
async def get_conversation_history(conversation_id: str):
    """Get conversation history"""
    if conversation_id not in rag_service.conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    memory = rag_service.conversations[conversation_id]
    return {
        "conversation_id": conversation_id,
        "history": memory.chat_memory.messages if hasattr(memory, 'chat_memory') else [],
        "timestamp": datetime.now()
    }

@app.delete("/conversations/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear conversation history"""
    if conversation_id in rag_service.conversations:
        del rag_service.conversations[conversation_id]
    
    return {
        "message": "Conversation cleared",
        "conversation_id": conversation_id,
        "timestamp": datetime.now()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)