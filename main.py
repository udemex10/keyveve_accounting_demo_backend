import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import os
import uuid
import shutil
import logging
import json
import asyncio
import datetime
import random
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt
import re

# Azure OpenAI and LangChain imports
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables and setup logging
load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# ------------------------------
# Azure OpenAI Configuration
# ------------------------------
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1")


@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
async def get_model_config():
    """Get Azure OpenAI model configuration with retry logic."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        logger.warning("AZURE_OPENAI_API_KEY not found in environment variables")
        raise ValueError("AZURE_OPENAI_API_KEY not found in environment variables")

    return {
        "azure_deployment": AZURE_OPENAI_MODEL,
        "openai_api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_key": api_key,
        "temperature": float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0.1")),
        "streaming": True  # Enable streaming
    }


@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
async def get_azure_openai_model():
    """Initialize Azure OpenAI model with retry logic."""
    try:
        model_config = await get_model_config()
        model = AzureChatOpenAI(**model_config)
        return model
    except Exception as e:
        logger.error(f"Error initializing Azure OpenAI model: {str(e)}")
        raise


# ------------------------------
# FastAPI App Setup
# ------------------------------
app = FastAPI(
    title="Keyveve AI Accounting API",
    description="API for AI-powered accounting workflow management",
    version="1.0.0",
)

# Allow local dev from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# In-Memory Data Structures
# ------------------------------
projects_data = {}  # key: project_id, value: dict with project data
messages_data = []  # store all messages in a list
doc_store = {}  # doc_id -> {file_name, doc_type, extracted_data}
project_counter = 1  # increments on new project creation
notifications = []  # store notifications
staff_members = []  # store staff member data
integrations = {
    "engagement_letter": {
        "name": "EngagementLetterApp",
        "connected": True,
        "last_sync": "2023-01-01T00:00:00Z"
    },
    "practice_management": {
        "name": "PracticeManagementSoftware",
        "connected": True,
        "last_sync": "2023-01-01T00:00:00Z"
    },
    "document_storage": {
        "name": "SharePoint",
        "connected": False,
        "last_sync": None
    },
    "cch_document_storage": {
        "name": "CCH Document Storage",
        "connected": True,
        "last_sync": "2023-01-15T00:00:00Z"
    },
    "keyveve_storage": {
        "name": "Keyveve Document Storage",
        "connected": True,
        "last_sync": "2023-01-20T00:00:00Z"
    },
    "outlook_calendar": {
        "name": "Outlook Calendar",
        "connected": True,
        "last_sync": "2023-01-15T00:00:00Z"
    },
    "google_calendar": {
        "name": "Google Calendar",
        "connected": False,
        "last_sync": None
    }
}


# ------------------------------
# Data Models
# ------------------------------
class ProjectCreate(BaseModel):
    client_name: str
    service_type: Optional[str] = None
    workflow_template: Optional[str] = None
    source: Optional[str] = "manual"  # manual, csv, pms (Practice Management Software)
    assigned_staff: Optional[List[str]] = None
    staff_roles: Optional[Dict[str, str]] = None


class StaffMember(BaseModel):
    id: str
    name: str
    email: str
    role: str
    avatar_url: Optional[str] = None


class Task(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, blocked
    deadline: Optional[str] = None
    assigned_to: Optional[List[str]] = None
    related_docs: Optional[List[str]] = None
    scheduled_start: Optional[str] = None
    scheduled_end: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())


class Project(BaseModel):
    id: int
    client_name: str
    status: str
    service_type: Optional[str]  # Changed from service_needed to service_type
    workflow_template: Optional[str] = None
    docs: List[Dict[str, Any]]
    tasks: List[Task]  # Changed from List[str] to List[Task]
    messages: List[Dict[str, Any]]
    assigned_staff: List[str] = []
    staff_roles: Optional[Dict[str, str]] = None
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())


class MessageCreate(BaseModel):
    project_id: int
    sender: str  # "staff" or "client"
    text: str
    sender_id: Optional[str] = None


class Message(BaseModel):
    id: str
    project_id: int
    sender: str
    text: str
    sender_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())


class QuestionRequest(BaseModel):
    project_id: Optional[int] = None
    question: str
    global_context: bool = False


class DocumentUploadResponse(BaseModel):
    doc_id: str
    doc_type: str
    extracted_data: str
    original_name: str
    stored_name: str
    storage_location: str = "cloud"  # cloud, sharepoint, cch, keyveve_storage
    doc_category: str = "client"  # client, internal, permanent
    status: str = "awaiting_review"


class DocumentUpdateRequest(BaseModel):
    doc_id: str
    doc_type: Optional[str] = None
    storage_location: Optional[str] = None
    doc_category: Optional[str] = None
    related_tasks: Optional[List[str]] = None
    final_name: Optional[str] = None
    status: Optional[str] = None


class PriceRequest(BaseModel):
    project_id: int
    complexity_factors: Optional[Dict[str, float]] = None


class PriceResponse(BaseModel):
    project_id: int
    suggested_price: float
    explanation: str


class IntegrationRequest(BaseModel):
    integration_type: str
    action: str = "connect"  # connect, disconnect, sync
    config: Optional[Dict[str, Any]] = None


class IntegrationResponse(BaseModel):
    integration_type: str
    status: str
    message: str
    connected: bool


class Notification(BaseModel):
    id: str
    project_id: int
    type: str  # reminder, alert, info
    message: str
    created_at: str
    read: bool = False


class ImportCSVRequest(BaseModel):
    file_content: str  # Base64 encoded or plain CSV


class ImportCSVResponse(BaseModel):
    success: bool
    message: str
    imported_count: int
    project_ids: List[int]


class TaskCreate(BaseModel):
    project_id: int
    title: str
    description: Optional[str] = None
    status: str = "pending"
    deadline: Optional[str] = None
    assigned_to: Optional[List[str]] = None
    related_docs: Optional[List[str]] = None


class TaskUpdate(BaseModel):
    task_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    deadline: Optional[str] = None
    assigned_to: Optional[List[str]] = None
    related_docs: Optional[List[str]] = None


class StaffAssignmentRequest(BaseModel):
    project_id: int
    staff_ids: List[str]
    staff_roles: Dict[str, str] = {}  # Maps staff_id to role ("point_of_contact", "partner_assigned", or "staff")


class TaskScheduleRequest(BaseModel):
    task_id: str
    staff_id: str
    scheduled_start: str
    scheduled_end: str
    sync_to_calendar: Optional[str] = None  # "outlook", "google", etc.


# ------------------------------
# Helper Functions
# ------------------------------
def generate_tasks_for_service_type(service_type: str, status: str, workflow_template: str = None) -> List[Task]:
    """Generate appropriate tasks based on service type, workflow template and project status."""
    # Basic tasks for each status
    status_tasks = {
        "Onboarding": [
            {"title": "Send welcome email to client", "status": "pending"},
            {"title": "Verify client identity and contact details", "status": "pending"},
            {"title": "Setup client in practice management system", "status": "pending"},
        ],
        "Docs Requested": [
            {"title": "Review document requirements", "status": "completed"},
            {"title": "Request required documents from client", "status": "completed"},
            {"title": "Send reminder in 7 days if no response", "status": "pending"},
        ],
        "Docs Received": [
            {"title": "Review received documents for completeness", "status": "pending"},
            {"title": "Categorize and organize documents", "status": "pending"},
            {"title": "Prepare for pricing analysis", "status": "pending"},
        ],
        "Pricing/Analysis": [
            {"title": "Analyze document complexity", "status": "pending"},
            {"title": "Calculate pricing based on document count and type", "status": "pending"},
            {"title": "Prepare engagement letter draft", "status": "pending"},
        ],
        "Awaiting Signature": [
            {"title": "Follow up on engagement letter if not signed within 3 days", "status": "pending"},
            {"title": "Verify signature authenticity", "status": "pending"},
            {"title": "Prepare onboarding documents", "status": "pending"},
        ],
        "Project Started": [
            {"title": "Assign staff to project tasks", "status": "pending"},
            {"title": "Create project timeline", "status": "pending"},
            {"title": "Schedule initial client consultation", "status": "pending"},
        ],
        "Completed": [
            {"title": "Send client satisfaction survey", "status": "pending"},
            {"title": "Archive project documents", "status": "pending"},
            {"title": "Schedule follow-up for next year", "status": "pending"},
        ]
    }

    # Additional service-specific tasks
    service_specific_tasks = {
        "Tax Return": [
            {"title": "Prepare tax workpapers", "status": "pending"},
            {"title": "Calculate tax liability", "status": "pending"},
            {"title": "Review deductions and credits", "status": "pending"},
            {"title": "Prepare final tax return", "status": "pending"},
            {"title": "Client review of tax return", "status": "pending"},
            {"title": "File tax return", "status": "pending"},
        ],
        "Bookkeeping": [
            {"title": "Set up chart of accounts", "status": "pending"},
            {"title": "Reconcile bank statements", "status": "pending"},
            {"title": "Process accounts receivable", "status": "pending"},
            {"title": "Process accounts payable", "status": "pending"},
            {"title": "Prepare monthly financial statements", "status": "pending"},
        ],
        "Audit": [
            {"title": "Perform risk assessment", "status": "pending"},
            {"title": "Test internal controls", "status": "pending"},
            {"title": "Perform substantive testing", "status": "pending"},
            {"title": "Review financial statements", "status": "pending"},
            {"title": "Draft audit report", "status": "pending"},
            {"title": "Present findings to client", "status": "pending"},
        ],
        "Advisory": [
            {"title": "Initial business assessment", "status": "pending"},
            {"title": "Financial projection modeling", "status": "pending"},
            {"title": "Strategic planning session", "status": "pending"},
            {"title": "Process improvement recommendations", "status": "pending"},
            {"title": "Present advisory findings", "status": "pending"},
        ],
        "Financial Planning": [
            {"title": "Gather client financial information", "status": "pending"},
            {"title": "Analyze current financial position", "status": "pending"},
            {"title": "Develop financial plan recommendations", "status": "pending"},
            {"title": "Present financial plan to client", "status": "pending"},
            {"title": "Implement financial plan", "status": "pending"},
        ],
    }

    # Enhanced service-specific tasks based on workflow template
    template_tasks = {}

    # Auto-assign cas-monthly template for Bookkeeping if no template specified
    if service_type == "Bookkeeping" and not workflow_template:
        workflow_template = "cas-monthly"

    # Tax-specific workflow templates
    if workflow_template == "tax-individual":
        template_tasks = {
            "Project Started": [
                {"title": "Review tax documents", "status": "pending"},
                {"title": "Prepare individual tax return", "status": "pending"},
                {"title": "Calculate tax liability", "status": "pending"},
                {"title": "Internal review of tax return", "status": "pending"},
                {"title": "Client review of tax return", "status": "pending"},
                {"title": "Prepare e-file documents", "status": "pending"},
                {"title": "File tax return", "status": "pending"},
            ]
        }
    elif workflow_template == "tax-business":
        template_tasks = {
            "Project Started": [
                {"title": "Review business financial documents", "status": "pending"},
                {"title": "Analyze business tax situation", "status": "pending"},
                {"title": "Prepare business tax return", "status": "pending"},
                {"title": "Calculate tax liability and credits", "status": "pending"},
                {"title": "Internal review of tax return", "status": "pending"},
                {"title": "Client review of tax return", "status": "pending"},
                {"title": "File business tax return", "status": "pending"},
            ]
        }
    # Audit-specific workflow templates
    elif workflow_template == "audit-standard":
        template_tasks = {
            "Project Started": [
                {"title": "Perform risk assessment", "status": "pending"},
                {"title": "Document internal controls", "status": "pending"},
                {"title": "Test internal controls", "status": "pending"},
                {"title": "Prepare sampling methodology", "status": "pending"},
                {"title": "Perform substantive testing", "status": "pending"},
                {"title": "Generate audit findings register", "status": "pending"},
                {"title": "Draft audit report", "status": "pending"},
                {"title": "Partner review of audit report", "status": "pending"},
                {"title": "Present findings to client", "status": "pending"},
            ]
        }
    # Bookkeeping & CAS workflow templates
    elif workflow_template == "cas-monthly":
        template_tasks = {
            "Project Started": [
                {"title": "Process monthly transactions", "status": "pending"},
                {"title": "Reconcile bank accounts", "status": "pending"},
                {"title": "Process accounts receivable", "status": "pending"},
                {"title": "Process accounts payable", "status": "pending"},
                {"title": "Generate monthly financial statements", "status": "pending"},
                {"title": "Client review meeting", "status": "pending"},
            ]
        }

    # Check if we should use template-specific tasks
    if status in ["Project Started", "Completed"] and workflow_template and status in template_tasks:
        base_tasks = status_tasks.get(status, [])
        service_tasks = template_tasks.get(status, [])

        tasks = []
        for task_dict in base_tasks + service_tasks:
            task_id = str(uuid.uuid4())
            task = Task(
                id=task_id,
                title=task_dict["title"],
                status=task_dict["status"],
                description=task_dict.get("description", ""),
                deadline=None,
                assigned_to=[],
                related_docs=[]
            )
            tasks.append(task)

        return tasks
    else:
        # For non-template workflows or other statuses, use the original logic
        # Only include service-specific tasks when the project has started
        if status in ["Project Started", "Completed"]:
            base_tasks = status_tasks.get(status, [])
            service_tasks = service_specific_tasks.get(service_type, [])

            # Convert dictionaries to Task objects
            tasks = []
            for task_dict in base_tasks + service_tasks:
                task_id = str(uuid.uuid4())
                task = Task(
                    id=task_id,
                    title=task_dict["title"],
                    status=task_dict["status"],
                    description=task_dict.get("description", ""),
                    deadline=None,
                    assigned_to=[],
                    related_docs=[]
                )
                tasks.append(task)

            return tasks
        else:
            # For earlier stages, just use the status-based tasks
            base_tasks = status_tasks.get(status, [])
            tasks = []
            for task_dict in base_tasks:
                task_id = str(uuid.uuid4())
                task = Task(
                    id=task_id,
                    title=task_dict["title"],
                    status=task_dict["status"],
                    description=task_dict.get("description", ""),
                    deadline=None,
                    assigned_to=[],
                    related_docs=[]
                )
                tasks.append(task)

            return tasks

async def process_document_with_ai(file_path: str, filename: str, client: str):
    """
    Process a document with Azure OpenAI and return structured metadata,
    including an AI‑suggested clean file‑name.

    The suggested name heuristic is:
        <client‑slug>_<year>_<doc_type_snake_case><ext>
    """
    try:
        chat_model = await get_azure_openai_model()

        prompt = PromptTemplate(
            template=(
                "You are a document classifier and information extractor for an accounting firm.\n"
                "The file name is {filename}. The client is {client}.\n"
                "Classify the document and extract key data.\n"
                "Return your analysis exactly as JSON:\n"
                "{{"
                "\"doc_type\":\"[Document Type]\","
                "\"extracted_data\":\"[Key]\","
                "\"doc_category\":\"[client|internal|permanent]\","
                "\"suggested_storage\":\"[cloud|sharepoint|cch_document_storage|keyveve_storage]\","
                "\"suggested_name\":\"[clean_file_name]\""
                "}}"
            ),
            input_variables=["filename", "client"],
        )

        chain = LLMChain(llm=chat_model, prompt=prompt)
        result = await chain.arun(filename=filename, client=client)

        # --- parse AI JSON --------------------------------------------------
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            parsed = {}

        # -------------------------------------------------------------------
        # fallback logic for any missing keys or complete JSON failure
        # -------------------------------------------------------------------
        doc_type = parsed.get("doc_type")
        if not doc_type:
            fn = filename.lower()
            if "w2" in fn or "w-2" in fn:
                doc_type = "W-2"
            elif "1099" in fn:
                doc_type = "1099"
            elif "invoice" in fn:
                doc_type = "Invoice"
            elif "bank" in fn and "statement" in fn:
                doc_type = "Bank Statement"
            else:
                doc_type = "Unknown"

        # heuristic for suggested file‑name
        year_match = re.search(r"(20\d{2})", filename)
        year = year_match.group(1) if year_match else datetime.datetime.now().year
        slug = re.sub(r"\s+", "_", client.strip().lower())
        ext = os.path.splitext(filename)[1] or ".pdf"
        fallback_name = f"{slug}_{year}_{doc_type.replace(' ', '_').lower()}{ext}"

        return {
            "doc_type": doc_type,
            "extracted_data": parsed.get("extracted_data", "No data extracted"),
            "doc_category": parsed.get("doc_category", "client"),
            "suggested_storage": parsed.get("suggested_storage", "cloud"),
            "suggested_name": parsed.get("suggested_name", fallback_name),
        }

    except Exception as e:
        logger.error(f"Error in AI document processing: {str(e)}")
        # final safe fallback
        return {
            "doc_type": "Unknown",
            "extracted_data": "Error in AI processing",
            "doc_category": "client",
            "suggested_storage": "cloud",
            "suggested_name": f"{re.sub(r'\\s+', '_', client.lower())}_{datetime.datetime.now().year}_unknown.pdf",
        }


# Background task for document processing
async def process_document_and_update_project(
    project_id: int,
    doc_id: str,
    file_path: str,
    original_filename: str
):
    """Background task: run AI processing and update in‑memory stores."""
    try:
        project = projects_data.get(project_id)
        if not project:
            logger.error(f"Project {project_id} not found while processing {doc_id}")
            return

        ai_result = await process_document_with_ai(
            file_path,
            original_filename,
            project["client_name"],
        )

        # ------------------------------------------------------------------
        # unpack AI result
        # ------------------------------------------------------------------
        doc_type = ai_result["doc_type"]
        extracted_data = ai_result["extracted_data"]
        doc_category = ai_result["doc_category"]
        storage_location = ai_result["suggested_storage"]
        suggested_name = ai_result["suggested_name"]

        # ------------------- update central document store ----------------
        if doc_id in doc_store:
            doc_store[doc_id].update(
                {
                    "doc_type": doc_type,
                    "extracted_data": extracted_data,
                    "doc_category": doc_category,
                    "storage_location": storage_location,
                    "suggested_name": suggested_name,
                    "status": "awaiting_review",
                }
            )

        # ------------------- update project‑embedded record ---------------
        for doc in project["docs"]:
            if doc["doc_id"] == doc_id:
                doc.update(
                    {
                        "doc_type": doc_type,
                        "extracted_data": extracted_data,
                        "doc_category": doc_category,
                        "storage_location": storage_location,
                        "suggested_name": suggested_name,
                        "status": "awaiting_review",
                    }
                )
                break

        # auto‑advance project status if appropriate
        if (
            project["status"] == "Docs Requested"
            and len(project["docs"]) > 0
        ):
            project["status"] = "Docs Received"
            project["tasks"] = [
                task.dict()
                for task in generate_tasks_for_service_type(
                    project.get("service_type", "Tax Return"),
                    "Docs Received",
                )
            ]

            notifications.append(
                {
                    "id": str(uuid.uuid4()),
                    "project_id": project_id,
                    "type": "info",
                    "message": f"All requested documents received for project #{project_id}. Ready for review.",
                    "created_at": datetime.datetime.now().isoformat(),
                    "read": False,
                }
            )

        logger.info(f"Document {doc_id} processed successfully: {doc_type}")

    except Exception as e:
        logger.error(f"Error in background processing for {doc_id}: {str(e)}")


def calculate_price(project_id: int, complexity_factors: Dict[str, float] = None) -> PriceResponse:
    """Calculate price based on project details and optional complexity factors."""
    if project_id not in projects_data:
        raise HTTPException(404, "Project not found")

    project = projects_data[project_id]

    # Base price based on service type
    base_prices = {
        "Tax Return": 300,
        "Bookkeeping": 500,
        "Audit": 1000,
        "Financial Planning": 750,
        "Advisory": 600
    }

    service = project.get("service_type", "Tax Return")
    base_price = base_prices.get(service, 400)

    # Adjust for document count
    doc_count = len(project.get("docs", []))
    doc_multiplier = 1 + (doc_count * 0.05)  # 5% increase per document

    # Adjust for document complexity
    complexity_sum = 0
    for doc in project.get("docs", []):
        doc_type = doc.get("doc_type", "Unknown")
        if doc_type == "W-2":
            complexity_sum += 0.1
        elif doc_type == "1099":
            complexity_sum += 0.15
        elif doc_type == "Balance Sheet" or doc_type == "Income Statement":
            complexity_sum += 0.2
        elif doc_type == "Bank Statement":
            complexity_sum += 0.1
        else:
            complexity_sum += 0.05

    # Apply optional custom complexity factors
    if complexity_factors:
        for factor, value in complexity_factors.items():
            complexity_sum += value

    # Calculate final price
    final_price = base_price * doc_multiplier * (1 + complexity_sum)
    final_price = round(final_price, 2)

    # Generate explanation
    explanation = (
        f"Base price for {service}: ${base_price}\n"
        f"Document count adjustment: {doc_multiplier:.2f}x\n"
        f"Complexity adjustment: {1 + complexity_sum:.2f}x\n"
        f"Final price: ${final_price}"
    )

    return PriceResponse(
        project_id=project_id,
        suggested_price=final_price,
        explanation=explanation
    )


def create_notification(project_id: int, type: str, message: str) -> Notification:
    """Create and store a notification."""
    notification_id = str(uuid.uuid4())
    notification = {
        "id": notification_id,
        "project_id": project_id,
        "type": type,
        "message": message,
        "created_at": datetime.datetime.now().isoformat(),
        "read": False
    }
    notifications.append(notification)
    return Notification(**notification)


def process_csv_import(csv_content: str) -> ImportCSVResponse:
    """Process CSV import and create projects."""
    global project_counter

    # For prototype, we'll simulate CSV parsing
    # In production, use pandas or csv module

    created_project_ids = []
    lines = csv_content.strip().split('\n')

    if len(lines) < 2:  # Need at least header + 1 data row
        return ImportCSVResponse(
            success=False,
            message="CSV file must contain header row and at least one data row",
            imported_count=0,
            project_ids=[]
        )

    header = lines[0].split(',')

    # Check for required columns
    required_cols = ["client_name", "service_type"]
    for col in required_cols:
        if col not in header:
            return ImportCSVResponse(
                success=False,
                message=f"CSV missing required column: {col}",
                imported_count=0,
                project_ids=[]
            )

    name_idx = header.index("client_name")
    service_idx = header.index("service_type")

    # Optional columns
    staff_idx = header.index("assigned_staff") if "assigned_staff" in header else -1

    # Process data rows
    for i in range(1, len(lines)):
        row = lines[i].split(',')
        if len(row) < len(header):
            continue  # Skip incomplete rows

        client_name = row[name_idx].strip()
        service_type = row[service_idx].strip()

        if not client_name:
            continue  # Skip rows without client name

        # Extract staff assignments if available
        assigned_staff = []
        if staff_idx >= 0 and staff_idx < len(row):
            staff = row[staff_idx].strip()
            if staff:
                assigned_staff = [s.strip() for s in staff.split(';')]

        # Create project
        pid = project_counter
        project_counter += 1

        projects_data[pid] = {
            "id": pid,
            "client_name": client_name,
            "service_type": service_type,
            "status": "Onboarding",
            "docs": [],
            "tasks": [task.dict() for task in generate_tasks_for_service_type(service_type, "Onboarding")],
            "messages": [],
            "assigned_staff": assigned_staff,
            "staff_roles": {},
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "import_source": "csv"
        }

        created_project_ids.append(pid)

    return ImportCSVResponse(
        success=True,
        message=f"Successfully imported {len(created_project_ids)} projects from CSV",
        imported_count=len(created_project_ids),
        project_ids=created_project_ids
    )


# ------------------------------
# Dummy Data Initialization
# ------------------------------
def init_dummy_data():
    """Initialize dummy data for the prototype."""
    global projects_data, doc_store, messages_data, project_counter, staff_members

    # Initialize staff members
    staff_members = [
        {
            "id": "staff-001",
            "name": "Sarah Johnson",
            "email": "sarah.johnson@keyveve.com",
            "role": "Senior Accountant",
            "avatar_url": "https://randomuser.me/api/portraits/women/20.jpg"
        },
        {
            "id": "staff-002",
            "name": "Michael Chen",
            "email": "michael.chen@keyveve.com",
            "role": "Tax Specialist",
            "avatar_url": "https://randomuser.me/api/portraits/men/32.jpg"
        },
        {
            "id": "staff-003",
            "name": "Jessica Rodriguez",
            "email": "jessica.rodriguez@keyveve.com",
            "role": "Audit Manager",
            "avatar_url": "https://randomuser.me/api/portraits/women/43.jpg"
        },
        {
            "id": "staff-004",
            "name": "David Kim",
            "email": "david.kim@keyveve.com",
            "role": "Bookkeeper",
            "avatar_url": "https://randomuser.me/api/portraits/men/55.jpg"
        }
    ]

    # Project #1: John Doe, "Docs Requested", Tax Return
    project1_tasks = generate_tasks_for_service_type("Tax Return", "Docs Requested")
    projects_data[1] = {
        "id": 1,
        "client_name": "John Doe",
        "service_type": "Tax Return",
        "status": "Docs Requested",
        "docs": [
            {
                "doc_id": "doc-abc1",
                "original_name": "prior_year_1040.pdf",
                "stored_name": "abc1.pdf",
                "doc_type": "1040 Return",
                "doc_category": "permanent",
                "storage_location": "cch_document_storage",
                "extracted_data": "Prior year taxes for John Doe showing AGI of $75,000"
            },
            # Add these enhanced documents for tax return client
            {
                "doc_id": "doc-abc2",
                "original_name": "john_doe_w2_2023.pdf",
                "stored_name": "abc2.pdf",
                "doc_type": "W-2",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "W-2 form showing wages of $82,500 and federal tax withheld of $16,500"
            },
            {
                "doc_id": "doc-abc3",
                "original_name": "john_doe_1099-int.pdf",
                "stored_name": "abc3.pdf",
                "doc_type": "1099-INT",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "Interest income of $1,250 from First National Bank"
            },
            {
                "doc_id": "doc-abc4",
                "original_name": "john_doe_mortgage_interest.pdf",
                "stored_name": "abc4.pdf",
                "doc_type": "1098",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "Mortgage interest paid: $12,400"
            },
            {
                "doc_id": "doc-abc5",
                "original_name": "john_doe_charitable_donations.pdf",
                "stored_name": "abc5.pdf",
                "doc_type": "Donation Receipt",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "Charitable donations totaling $5,600 to various organizations"
            },
            {
                "doc_id": "doc-abc6",
                "original_name": "tax_planning_workpaper.xlsx",
                "stored_name": "abc6.xlsx",
                "doc_type": "Workpaper",
                "doc_category": "internal",
                "storage_location": "sharepoint",
                "extracted_data": "Internal tax planning calculations and scenarios"
            }
        ],
        "tasks": [task.dict() for task in project1_tasks],
        "messages": [
            {
                "id": "msg-111",
                "project_id": 1,
                "sender": "staff",
                "sender_id": "staff-002",
                "text": "Hello John, please upload your W-2 for this year.",
                "timestamp": (datetime.datetime.now() - datetime.timedelta(days=2)).isoformat()
            }
        ],
        "assigned_staff": ["staff-002"],
        "staff_roles": {
            "staff-002": "point_of_contact"
        },
        "created_at": (datetime.datetime.now() - datetime.timedelta(days=5)).isoformat(),
        "updated_at": (datetime.datetime.now() - datetime.timedelta(days=2)).isoformat()
    }

    # Project #2: Acme Corp, "Docs Received", Corporate Tax
    project2_tasks = generate_tasks_for_service_type("Tax Return", "Docs Received")
    # Update related docs for tasks
    project2_tasks[0].related_docs = ["doc-xyz2", "doc-xyz3"]
    project2_tasks[1].related_docs = ["doc-xyz2", "doc-xyz3"]
    projects_data[2] = {
        "id": 2,
        "client_name": "Acme Corp",
        "service_type": "Tax Return",
        "status": "Docs Received",
        "docs": [
            {
                "doc_id": "doc-xyz2",
                "original_name": "acme_2023_invoice.pdf",
                "stored_name": "xyz2.pdf",
                "doc_type": "Invoice",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "Invoice #2023 from vendor XYZ for $12,500"
            },
            {
                "doc_id": "doc-xyz3",
                "original_name": "acme_w-9.pdf",
                "stored_name": "xyz3.pdf",
                "doc_type": "W-9",
                "doc_category": "permanent",
                "storage_location": "cch_document_storage",
                "extracted_data": "W-9 form with EIN 12-3456789"
            }
        ],
        "tasks": [task.dict() for task in project2_tasks],
        "messages": [
            {
                "id": "msg-222",
                "project_id": 2,
                "sender": "client",
                "text": "We have uploaded the invoices and W-9 form. Please let us know next steps.",
                "timestamp": (datetime.datetime.now() - datetime.timedelta(days=1)).isoformat()
            },
            {
                "id": "msg-223",
                "project_id": 2,
                "sender": "staff",
                "sender_id": "staff-002",
                "text": "Thanks Acme, we will review and update you soon.",
                "timestamp": datetime.datetime.now().isoformat()
            }
        ],
        "assigned_staff": ["staff-002", "staff-001"],
        "staff_roles": {
            "staff-002": "point_of_contact",
            "staff-001": "partner_assigned"
        },
        "created_at": (datetime.datetime.now() - datetime.timedelta(days=7)).isoformat(),
        "updated_at": datetime.datetime.now().isoformat()
    }

    # Project #3: Jane Smith, "Awaiting Signature", Audit
    project3_tasks = generate_tasks_for_service_type("Audit", "Awaiting Signature")
    projects_data[3] = {
        "id": 3,
        "client_name": "Jane Smith",
        "service_type": "Audit",
        "status": "Awaiting Signature",
        "docs": [
            {
                "doc_id": "doc-jane1",
                "original_name": "bank_statement_jan.pdf",
                "stored_name": "jane1.pdf",
                "doc_type": "Bank Statement",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "January statement with balance of $45,200"
            },
            {
                "doc_id": "doc-jane2",
                "original_name": "balance_sheet_q4.pdf",
                "stored_name": "jane2.pdf",
                "doc_type": "Balance Sheet",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "Q4 balance sheet with assets of $230,500"
            },
            {
                "doc_id": "doc-jane3",
                "original_name": "audit_planning_memo.pdf",
                "stored_name": "jane3.pdf",
                "doc_type": "Audit Planning Memo",
                "doc_category": "internal",
                "storage_location": "sharepoint",
                "extracted_data": "Internal planning document for Jane Smith audit"
            },
            # Add these enhanced documents for audit client
            {
                "doc_id": "doc-jane4",
                "original_name": "income_statement_q4.pdf",
                "stored_name": "jane4.pdf",
                "doc_type": "Income Statement",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "Q4 income statement showing revenue of $580,000 and net income of $145,000"
            },
            {
                "doc_id": "doc-jane5",
                "original_name": "cash_flow_statement.pdf",
                "stored_name": "jane5.pdf",
                "doc_type": "Cash Flow Statement",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "Annual cash flow statement with operating cash flow of $210,000"
            },
            {
                "doc_id": "doc-jane6",
                "original_name": "inventory_listing.xlsx",
                "stored_name": "jane6.xlsx",
                "doc_type": "Inventory Report",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "Detailed inventory listing with 1,250 items valued at $320,000"
            },
            {
                "doc_id": "doc-jane7",
                "original_name": "accounts_receivable_aging.xlsx",
                "stored_name": "jane7.xlsx",
                "doc_type": "Accounts Receivable Aging",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "A/R aging report showing $125,000 in receivables with $15,000 over 90 days"
            },
            {
                "doc_id": "doc-jane8",
                "original_name": "audit_risk_assessment.docx",
                "stored_name": "jane8.docx",
                "doc_type": "Audit Risk Assessment",
                "doc_category": "internal",
                "storage_location": "sharepoint",
                "extracted_data": "Internal audit risk assessment documenting key risk areas"
            },
            {
                "doc_id": "doc-jane9",
                "original_name": "audit_sampling_workpaper.xlsx",
                "stored_name": "jane9.xlsx",
                "doc_type": "Audit Sampling Workpaper",
                "doc_category": "internal",
                "storage_location": "sharepoint",
                "extracted_data": "Documentation of sampling methodology and selected transactions"
            }
        ],
        "tasks": [task.dict() for task in project3_tasks],
        "messages": [
            {
                "id": "msg-333",
                "project_id": 3,
                "sender": "staff",
                "sender_id": "staff-003",
                "text": "Jane, your documents look good. We'll do a quick review and send an engagement letter.",
                "timestamp": (datetime.datetime.now() - datetime.timedelta(days=2)).isoformat()
            },
            {
                "id": "msg-334",
                "project_id": 3,
                "sender": "staff",
                "sender_id": "staff-003",
                "text": "The engagement letter has been sent. Please sign at your earliest convenience.",
                "timestamp": (datetime.datetime.now() - datetime.timedelta(hours=4)).isoformat()
            }
        ],
        "assigned_staff": ["staff-003"],
        "staff_roles": {
            "staff-003": "partner_assigned"
        },
        "created_at": (datetime.datetime.now() - datetime.timedelta(days=10)).isoformat(),
        "updated_at": (datetime.datetime.now() - datetime.timedelta(hours=4)).isoformat()
    }

    # Project #4: Tech Startup Inc, "Project Started", Bookkeeping
    project4_tasks = generate_tasks_for_service_type("Bookkeeping", "Project Started")
    projects_data[4] = {
        "id": 4,
        "client_name": "Tech Startup Inc",
        "service_type": "Bookkeeping",
        "status": "Project Started",
        "docs": [
            {
                "doc_id": "doc-tech1",
                "original_name": "q1_expenses.pdf",
                "stored_name": "tech1.pdf",
                "doc_type": "Expense Report",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "Q1 expenses totaling $128,750"
            },
            {
                "doc_id": "doc-tech2",
                "original_name": "q1_revenue.pdf",
                "stored_name": "tech2.pdf",
                "doc_type": "Revenue Report",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "Q1 revenue of $215,300"
            },
            {
                "doc_id": "doc-tech3",
                "original_name": "chart_of_accounts.xlsx",
                "stored_name": "tech3.xlsx",
                "doc_type": "Chart of Accounts",
                "doc_category": "permanent",
                "storage_location": "sharepoint",
                "extracted_data": "Company chart of accounts structure"
            },
            {
                "doc_id": "doc-tech4",
                "original_name": "bookkeeping_workpaper.xlsx",
                "stored_name": "tech4.xlsx",
                "doc_type": "Workpaper",
                "doc_category": "internal",
                "storage_location": "sharepoint",
                "extracted_data": "Internal workpaper for reconciliation"
            },
            # Add these enhanced documents for bookkeeping client
            {
                "doc_id": "doc-tech5",
                "original_name": "q1_bank_statement_chase.pdf",
                "stored_name": "tech5.pdf",
                "doc_type": "Bank Statement",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "Q1 checking account statement with ending balance of $45,230"
            },
            {
                "doc_id": "doc-tech6",
                "original_name": "q1_bank_statement_wells.pdf",
                "stored_name": "tech6.pdf",
                "doc_type": "Bank Statement",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "Q1 savings account statement with ending balance of $120,500"
            },
            {
                "doc_id": "doc-tech7",
                "original_name": "q1_credit_card_statement.pdf",
                "stored_name": "tech7.pdf",
                "doc_type": "Credit Card Statement",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "Q1 business credit card statement with $28,450 in expenses"
            },
            {
                "doc_id": "doc-tech8",
                "original_name": "accounts_payable_aging.xlsx",
                "stored_name": "tech8.xlsx",
                "doc_type": "Accounts Payable Aging",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "A/P aging report showing $62,300 in payables with $8,500 over 60 days"
            },
            {
                "doc_id": "doc-tech9",
                "original_name": "payroll_summary_q1.pdf",
                "stored_name": "tech9.pdf",
                "doc_type": "Payroll Summary",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "Q1 payroll summary showing $235,000 in wages paid to 15 employees"
            },
            {
                "doc_id": "doc-tech10",
                "original_name": "bank_reconciliation.xlsx",
                "stored_name": "tech10.xlsx",
                "doc_type": "Bank Reconciliation",
                "doc_category": "internal",
                "storage_location": "sharepoint",
                "extracted_data": "Internal bank reconciliation workpaper for checking account"
            },
            {
                "doc_id": "doc-tech11",
                "original_name": "depreciation_schedule.xlsx",
                "stored_name": "tech11.xlsx",
                "doc_type": "Depreciation Schedule",
                "doc_category": "internal",
                "storage_location": "sharepoint",
                "extracted_data": "Fixed asset depreciation schedule for the current year"
            }
        ],
        "tasks": [task.dict() for task in project4_tasks],
        "messages": [
            {
                "id": "msg-444",
                "project_id": 4,
                "sender": "client",
                "text": "Looking forward to working with your team on our bookkeeping needs.",
                "timestamp": (datetime.datetime.now() - datetime.timedelta(days=5)).isoformat()
            },
            {
                "id": "msg-445",
                "project_id": 4,
                "sender": "staff",
                "sender_id": "staff-004",
                "text": "We've set up your account in our system and will begin organizing your financials this week.",
                "timestamp": (datetime.datetime.now() - datetime.timedelta(days=3)).isoformat()
            }
        ],
        "assigned_staff": ["staff-004", "staff-001"],
        "staff_roles": {
            "staff-004": "point_of_contact",
            "staff-001": "partner_assigned"
        },
        "created_at": (datetime.datetime.now() - datetime.timedelta(days=15)).isoformat(),
        "updated_at": (datetime.datetime.now() - datetime.timedelta(days=3)).isoformat()
    }

    # Project #5: Smith Family LLC, "Project Started", Financial Planning
    project5_tasks = generate_tasks_for_service_type("Financial Planning", "Project Started")
    projects_data[5] = {
        "id": 5,
        "client_name": "Smith Family LLC",
        "service_type": "Financial Planning",
        "status": "Project Started",
        "docs": [
            {
                "doc_id": "doc-smith1",
                "original_name": "investment_portfolio.pdf",
                "stored_name": "smith1.pdf",
                "doc_type": "Investment Portfolio",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "Current investment portfolio valued at $1.2M"
            },
            {
                "doc_id": "doc-smith2",
                "original_name": "retirement_accounts.pdf",
                "stored_name": "smith2.pdf",
                "doc_type": "Retirement Accounts",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "Summary of 401k, IRA and other retirement accounts"
            },
            {
                "doc_id": "doc-smith3",
                "original_name": "financial_plan_draft.docx",
                "stored_name": "smith3.docx",
                "doc_type": "Financial Plan Draft",
                "doc_category": "internal",
                "storage_location": "sharepoint",
                "extracted_data": "Draft financial plan with investment recommendations"
            }
        ],
        "tasks": [task.dict() for task in project5_tasks],
        "messages": [
            {
                "id": "msg-556",
                "project_id": 5,
                "sender": "client",
                "text": "We'd like to focus on retirement planning and college savings for our children.",
                "timestamp": (datetime.datetime.now() - datetime.timedelta(days=7)).isoformat()
            },
            {
                "id": "msg-557",
                "project_id": 5,
                "sender": "staff",
                "sender_id": "staff-001",
                "text": "Thanks for sharing your priorities. We'll incorporate these into your financial plan.",
                "timestamp": (datetime.datetime.now() - datetime.timedelta(days=6)).isoformat()
            }
        ],
        "assigned_staff": ["staff-001"],
        "staff_roles": {
            "staff-001": "partner_assigned"
        },
        "created_at": (datetime.datetime.now() - datetime.timedelta(days=12)).isoformat(),
        "updated_at": (datetime.datetime.now() - datetime.timedelta(days=3)).isoformat()
    }

    # Project #6: Metro Hospital, "Pricing/Analysis", Advisory
    project6_tasks = generate_tasks_for_service_type("Advisory", "Pricing/Analysis")
    projects_data[6] = {
        "id": 6,
        "client_name": "Metro Hospital",
        "service_type": "Advisory",
        "status": "Pricing/Analysis",
        "docs": [
            {
                "doc_id": "doc-metro1",
                "original_name": "financial_statements_2023.pdf",
                "stored_name": "metro1.pdf",
                "doc_type": "Financial Statements",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "Annual financial statements showing revenue of $45M"
            },
            {
                "doc_id": "doc-metro2",
                "original_name": "operational_metrics.xlsx",
                "stored_name": "metro2.xlsx",
                "doc_type": "Operational Metrics",
                "doc_category": "client",
                "storage_location": "cloud",
                "extracted_data": "Key performance indicators for hospital operations"
            },
            {
                "doc_id": "doc-metro3",
                "original_name": "advisory_proposal_draft.docx",
                "stored_name": "metro3.docx",
                "doc_type": "Advisory Proposal",
                "doc_category": "internal",
                "storage_location": "sharepoint",
                "extracted_data": "Draft proposal for operational efficiency consulting"
            }
        ],
        "tasks": [task.dict() for task in project6_tasks],
        "messages": [
            {
                "id": "msg-668",
                "project_id": 6,
                "sender": "client",
                "text": "We're particularly interested in improving our billing efficiency and reducing operational costs.",
                "timestamp": (datetime.datetime.now() - datetime.timedelta(days=4)).isoformat()
            },
            {
                "id": "msg-669",
                "project_id": 6,
                "sender": "staff",
                "sender_id": "staff-003",
                "text": "We're reviewing your financials now and will prepare a customized proposal addressing these areas.",
                "timestamp": (datetime.datetime.now() - datetime.timedelta(days=3)).isoformat()
            }
        ],
        "assigned_staff": ["staff-003", "staff-001"],
        "staff_roles": {
            "staff-003": "point_of_contact",
            "staff-001": "partner_assigned"
        },
        "created_at": (datetime.datetime.now() - datetime.timedelta(days=8)).isoformat(),
        "updated_at": (datetime.datetime.now() - datetime.timedelta(days=1)).isoformat()
    }

    # Add some notifications
    notifications.extend([
        {
            "id": str(uuid.uuid4()),
            "project_id": 1,
            "type": "reminder",
            "message": "Client has not uploaded W-2 within 3 days of request",
            "created_at": datetime.datetime.now().isoformat(),
            "read": False
        },
        {
            "id": str(uuid.uuid4()),
            "project_id": 3,
            "type": "info",
            "message": "Engagement letter sent and awaiting signature",
            "created_at": (datetime.datetime.now() - datetime.timedelta(hours=4)).isoformat(),
            "read": True
        },
        {
            "id": str(uuid.uuid4()),
            "project_id": 5,
            "type": "info",
            "message": "New document uploaded by client: Investment Portfolio",
            "created_at": (datetime.datetime.now() - datetime.timedelta(hours=8)).isoformat(),
            "read": False
        },
        {
            "id": str(uuid.uuid4()),
            "project_id": 6,
            "type": "alert",
            "message": "Advisory proposal approval needed before proceeding",
            "created_at": (datetime.datetime.now() - datetime.timedelta(hours=2)).isoformat(),
            "read": False
        }
    ])

    # Store doc records
    for project in projects_data.values():
        for doc in project["docs"]:
            doc_id = doc["doc_id"]
            doc_store[doc_id] = {
                "file_name": doc["stored_name"],
                "doc_type": doc["doc_type"],
                "extracted_data": doc["extracted_data"],
                "doc_category": doc.get("doc_category", "client"),
                "storage_location": doc.get("storage_location", "cloud")
            }

    # Next project ID should be 7
    project_counter = 7


# ------------------------------
# API Routes
# ------------------------------
@app.get("/")
def root():
    """Root endpoint that returns basic API information."""
    return {
        "message": "Keyveve AI Accounting API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "projects": "/projects/",
            "documents": "/documents/upload",
            "messages": "/messages/",
            "qa": "/qa",
            "pricing": "/pricing/",
            "integrations": "/integrations/",
            "staff": "/staff/"
        }
    }


@app.get("/projects/", response_model=List[Project])
def list_projects(
        status: Optional[str] = None,
        service_type: Optional[str] = None,
        staff_id: Optional[str] = None,
        limit: int = Query(100, gt=0, le=100)
):
    """
    Get a list of projects with optional filtering.

    - Filter by status (e.g., 'Onboarding', 'Docs Requested', etc.)
    - Filter by service type (e.g., 'Tax Return', 'Bookkeeping', etc.)
    - Filter by assigned staff member
    - Limit number of results returned
    """
    results = []

    for pid, project in projects_data.items():
        # Apply filters
        if status and project["status"] != status:
            continue
        if service_type and project.get("service_type") != service_type:
            continue
        if staff_id and staff_id not in project.get("assigned_staff", []):
            continue

        results.append(Project(
            id=project["id"],
            client_name=project["client_name"],
            status=project["status"],
            service_type=project.get("service_type"),
            docs=project["docs"],
            tasks=[Task(**task) if isinstance(task, dict) else task for task in project["tasks"]],
            messages=project["messages"],
            assigned_staff=project.get("assigned_staff", []),
            staff_roles=project.get("staff_roles", {}),
            created_at=project.get("created_at", datetime.datetime.now().isoformat()),
            updated_at=project.get("updated_at", datetime.datetime.now().isoformat())
        ))

        if len(results) >= limit:
            break

    return results


@app.get("/projects/{project_id}", response_model=Project)
def get_project(project_id: int):
    """Get details for a specific project by ID."""
    if project_id not in projects_data:
        raise HTTPException(404, "Project not found")

    p = projects_data[project_id]
    return Project(
        id=p["id"],
        client_name=p["client_name"],
        status=p["status"],
        service_type=p["service_type"],
        docs=p["docs"],
        tasks=[Task(**task) if isinstance(task, dict) else task for task in p["tasks"]],
        messages=p["messages"],
        assigned_staff=p.get("assigned_staff", []),
        staff_roles=p.get("staff_roles", {}),
        created_at=p.get("created_at", datetime.datetime.now().isoformat()),
        updated_at=p.get("updated_at", datetime.datetime.now().isoformat())
    )


@app.post("/projects/", response_model=Project)
async def create_project(proj_in: ProjectCreate, background_tasks: BackgroundTasks):
    """
    Create a new project for a client with specified workflow template.
    """
    global project_counter
    pid = project_counter
    project_counter += 1

    status = "Onboarding"
    workflow_template = proj_in.workflow_template if hasattr(proj_in, 'workflow_template') else None

    # Generate tasks based on service type and optionally workflow template
    tasks = generate_tasks_for_service_type(
        proj_in.service_type or "Tax Return",
        status,
        workflow_template
    )

    timestamp = datetime.datetime.now().isoformat()

    projects_data[pid] = {
        "id": pid,
        "client_name": proj_in.client_name,
        "service_type": proj_in.service_type or "Tax Return",
        "workflow_template": workflow_template,
        "status": status,
        "docs": [],
        "tasks": [task.dict() for task in tasks],
        "messages": [],
        "assigned_staff": proj_in.assigned_staff or [],
        "staff_roles": proj_in.staff_roles or {},
        "created_at": timestamp,
        "updated_at": timestamp,
        "source": proj_in.source
    }

    # Create welcome notification
    background_tasks.add_task(
        create_notification,
        pid,
        "info",
        f"New {proj_in.service_type or 'Tax Return'} project created for {proj_in.client_name}"
    )

    return Project(
        id=pid,
        client_name=projects_data[pid]["client_name"],
        status=projects_data[pid]["status"],
        service_type=projects_data[pid]["service_type"],
        workflow_template=workflow_template,
        docs=projects_data[pid]["docs"],
        tasks=[Task(**task) for task in projects_data[pid]["tasks"]],
        messages=projects_data[pid]["messages"],
        assigned_staff=projects_data[pid]["assigned_staff"],
        staff_roles=projects_data[pid].get("staff_roles", {}),
        created_at=projects_data[pid]["created_at"],
        updated_at=projects_data[pid]["updated_at"]
    )


@app.post("/calendar/sync", response_model=Dict[str, Any])
async def sync_to_calendar(
        task_id: str,
        calendar_type: str = "outlook",  # outlook or google
        start_date: str = None,
        end_date: str = None,
):
    """
    Sync a task to calendar (Outlook or Google).
    This is a mock implementation for the prototype.
    """
    # Find the task
    task_found = False
    for project in projects_data.values():
        for task in project.get("tasks", []):
            if isinstance(task, dict) and task.get("id") == task_id:
                task_found = True
                break
        if task_found:
            break

    if not task_found:
        raise HTTPException(404, "Task not found")

    # Mock successful response
    return {
        "success": True,
        "task_id": task_id,
        "calendar_type": calendar_type,
        "message": f"Task successfully synced to {calendar_type} calendar"
    }

@app.patch("/projects/{project_id}/status")
async def update_project_status(
        project_id: int,
        new_status: str,
        background_tasks: BackgroundTasks
):
    """
    Update the status of a project.

    - Stages: 'Onboarding', 'Docs Requested', 'Docs Received', 'Pricing/Analysis',
              'Awaiting Signature', 'Project Started', 'Completed'
    - Automatically generates new tasks appropriate for the new status
    - Creates a notification about the status change
    """
    if project_id not in projects_data:
        raise HTTPException(404, "Project not found")

    # Validate status is one of the expected values
    valid_statuses = [
        "Onboarding", "Docs Requested", "Docs Received",
        "Pricing/Analysis", "Awaiting Signature", "Project Started", "Completed"
    ]

    if new_status not in valid_statuses:
        raise HTTPException(400, f"Invalid status. Must be one of: {', '.join(valid_statuses)}")

    old_status = projects_data[project_id]["status"]

    # Update status and timestamp
    projects_data[project_id]["status"] = new_status
    projects_data[project_id]["updated_at"] = datetime.datetime.now().isoformat()

    # Generate new tasks appropriate for the status
    service_type = projects_data[project_id].get("service_type")
    new_tasks = generate_tasks_for_service_type(service_type, new_status)
    projects_data[project_id]["tasks"] = [task.dict() for task in new_tasks]

    # Create notification about status change
    if old_status != new_status:
        notification_message = f"Project status changed from '{old_status}' to '{new_status}'"
        background_tasks.add_task(
            create_notification,
            project_id,
            "info",
            notification_message
        )

    return {
        "success": True,
        "status": new_status,
        "project_id": project_id,
        "tasks": [task.dict() for task in new_tasks]
    }


@app.post("/projects/{project_id}/assign-staff")
async def assign_staff_to_project(
        project_id: int,
        assignment: StaffAssignmentRequest,
        background_tasks: BackgroundTasks
):
    """
    Assign staff members to a project.

    - Can assign multiple staff members with specific roles
    - Updates project record with staff IDs and roles
    - Creates a notification
    """
    if project_id not in projects_data:
        raise HTTPException(404, "Project not found")

    # Update assigned staff list
    projects_data[project_id]["assigned_staff"] = assignment.staff_ids

    # Update staff roles
    if not "staff_roles" in projects_data[project_id]:
        projects_data[project_id]["staff_roles"] = {}

    for staff_id, role in assignment.staff_roles.items():
        if staff_id in assignment.staff_ids:
            projects_data[project_id]["staff_roles"][staff_id] = role

    projects_data[project_id]["updated_at"] = datetime.datetime.now().isoformat()

    # Create notification
    staff_names = []
    for staff_id in assignment.staff_ids:
        for staff in staff_members:
            if staff["id"] == staff_id:
                role_text = ""
                if staff_id in assignment.staff_roles:
                    if assignment.staff_roles[staff_id] == "point_of_contact":
                        role_text = " (Point of Contact)"
                    elif assignment.staff_roles[staff_id] == "partner_assigned":
                        role_text = " (Partner Assigned)"
                staff_names.append(f"{staff['name']}{role_text}")
                break

    staff_names_str = ", ".join(staff_names) if staff_names else "New staff"
    notification_message = f"{staff_names_str} assigned to project #{project_id}"

    background_tasks.add_task(
        create_notification,
        project_id,
        "info",
        notification_message
    )

    return {
        "success": True,
        "project_id": project_id,
        "assigned_staff": assignment.staff_ids,
        "staff_roles": assignment.staff_roles
    }



@app.post("/tasks/", response_model=Task)
async def create_task(task_in: TaskCreate, background_tasks: BackgroundTasks):
    """
    Create a new task for a project.

    - Tasks can be assigned to staff members
    - Tasks can be associated with documents
    """
    if task_in.project_id not in projects_data:
        raise HTTPException(404, "Project not found")

    # Create task ID
    task_id = str(uuid.uuid4())

    # Create task object
    task = Task(
        id=task_id,
        title=task_in.title,
        description=task_in.description,
        status=task_in.status,
        deadline=task_in.deadline,
        assigned_to=task_in.assigned_to or [],
        related_docs=task_in.related_docs or [],
        created_at=datetime.datetime.now().isoformat(),
        updated_at=datetime.datetime.now().isoformat()
    )

    # Add to project's task list
    projects_data[task_in.project_id]["tasks"].append(task.dict())
    projects_data[task_in.project_id]["updated_at"] = datetime.datetime.now().isoformat()

    # Create notification if task is assigned
    if task.assigned_to:
        staff_names = []
        for staff_id in task.assigned_to:
            for staff in staff_members:
                if staff["id"] == staff_id:
                    staff_names.append(staff["name"])
                    break

        assignees = ", ".join(staff_names) if staff_names else "Staff"
        notification_message = f"Task '{task.title}' assigned to {assignees}"

        background_tasks.add_task(
            create_notification,
            task_in.project_id,
            "info",
            notification_message
        )

    return task


@app.patch("/tasks/{task_id}")
async def update_task(
        task_id: str,
        update: TaskUpdate,
        background_tasks: BackgroundTasks
):
    """
    Update an existing task.

    - Can update task details, status, assignments, etc.
    """
    # Find the task in all projects
    for project_id, project in projects_data.items():
        for i, task in enumerate(project["tasks"]):
            if isinstance(task, dict) and task.get("id") == task_id:
                # Update task fields
                if update.title is not None:
                    project["tasks"][i]["title"] = update.title
                if update.description is not None:
                    project["tasks"][i]["description"] = update.description
                if update.status is not None:
                    old_status = project["tasks"][i].get("status")
                    project["tasks"][i]["status"] = update.status

                    # Create notification for status change
                    if old_status != update.status and update.status == "completed":
                        notification_message = f"Task '{project['tasks'][i]['title']}' marked as completed"
                        background_tasks.add_task(
                            create_notification,
                            project_id,
                            "info",
                            notification_message
                        )

                if update.deadline is not None:
                    project["tasks"][i]["deadline"] = update.deadline
                if update.assigned_to is not None:
                    project["tasks"][i]["assigned_to"] = update.assigned_to
                if update.related_docs is not None:
                    project["tasks"][i]["related_docs"] = update.related_docs

                project["tasks"][i]["updated_at"] = datetime.datetime.now().isoformat()
                project["updated_at"] = datetime.datetime.now().isoformat()

                return {
                    "success": True,
                    "task_id": task_id,
                    "task": project["tasks"][i]
                }

    raise HTTPException(404, "Task not found")


@app.post("/tasks/{task_id}/schedule")
async def schedule_task(
        task_id: str,
        schedule: TaskScheduleRequest,
        background_tasks: BackgroundTasks
):
    """
    Schedule a task for a specific staff member.

    - Can specify start and end dates
    - Can sync to calendar (Outlook or Google)
    - Assigns the task to the staff member
    """
    # Find the task in all projects
    for project_id, project in projects_data.items():
        for i, task in enumerate(project["tasks"]):
            if isinstance(task, dict) and task.get("id") == task_id:
                # Update task scheduling
                project["tasks"][i]["scheduled_start"] = schedule.scheduled_start
                project["tasks"][i]["scheduled_end"] = schedule.scheduled_end
                project["tasks"][i]["assigned_to"] = [schedule.staff_id]
                project["tasks"][i]["updated_at"] = datetime.datetime.now().isoformat()

                # Create notification
                staff_name = "Unknown"
                for staff in staff_members:
                    if staff["id"] == schedule.staff_id:
                        staff_name = staff["name"]
                        break

                sync_text = ""
                if schedule.sync_to_calendar:
                    sync_text = f" and synced to {schedule.sync_to_calendar.capitalize()} Calendar"

                notification_message = f"Task '{project['tasks'][i]['title']}' scheduled for {staff_name}{sync_text}"

                background_tasks.add_task(
                    create_notification,
                    project_id,
                    "info",
                    notification_message
                )

                return {
                    "success": True,
                    "task_id": task_id,
                    "scheduled": True,
                    "calendar_synced": bool(schedule.sync_to_calendar),
                    "calendar_type": schedule.sync_to_calendar
                }

    raise HTTPException(404, "Task not found")


@app.get("/staff/", response_model=List[StaffMember])
def list_staff():
    """Get a list of all staff members."""
    return [StaffMember(**staff) for staff in staff_members]


@app.get("/staff/{staff_id}", response_model=StaffMember)
def get_staff(staff_id: str):
    """Get details for a specific staff member."""
    for staff in staff_members:
        if staff["id"] == staff_id:
            return StaffMember(**staff)

    raise HTTPException(404, "Staff member not found")


@app.post("/messages/", response_model=Message)
async def create_message(msg_in: MessageCreate, background_tasks: BackgroundTasks):
    """
    Create a new message for a project.

    - Messages can be from staff or client
    - Messages are stored with the project for easy reference
    - Creates a notification for the recipient
    """
    if msg_in.project_id not in projects_data:
        raise HTTPException(404, "Project not found")

    # Validate sender
    if msg_in.sender not in ["staff", "client"]:
        raise HTTPException(400, "Sender must be either 'staff' or 'client'")

    msg_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()

    msg_obj = {
        "id": msg_id,
        "project_id": msg_in.project_id,
        "sender": msg_in.sender,
        "sender_id": msg_in.sender_id,
        "text": msg_in.text,
        "timestamp": timestamp
    }

    # Add to project messages and global messages list
    projects_data[msg_in.project_id]["messages"].append(msg_obj)
    messages_data.append(msg_obj)

    # Update project timestamp
    projects_data[msg_in.project_id]["updated_at"] = timestamp

    # Create notification for recipient
    recipient = "staff" if msg_in.sender == "client" else "client"

    # Add staff name if available
    sender_name = ""
    if msg_in.sender == "staff" and msg_in.sender_id:
        for staff in staff_members:
            if staff["id"] == msg_in.sender_id:
                sender_name = f" from {staff['name']}"
                break

    notification_message = f"New message{sender_name} in project #{msg_in.project_id}: {msg_in.text[:50]}..."

    background_tasks.add_task(
        create_notification,
        msg_in.project_id,
        "info",
        notification_message
    )

    return Message(**msg_obj)


@app.get("/messages/{project_id}", response_model=List[Message])
def get_project_messages(project_id: int, limit: int = Query(50, gt=0, le=100)):
    """Get messages for a specific project."""
    if project_id not in projects_data:
        raise HTTPException(404, "Project not found")

    messages = [Message(**msg) for msg in projects_data[project_id]["messages"]]

    # Sort by timestamp, newest first
    messages.sort(key=lambda x: x.timestamp, reverse=True)

    return messages[:limit]


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
        background_tasks: BackgroundTasks,
        project_id: int = Form(...),
        file: UploadFile = File(...),
        process_async: bool = Form(True),
        storage_location: str = Form("cloud"),
        doc_category: str = Form("client")
):
    """
    Upload and process a document for a project.

    - Client or staff can upload documents
    - Documents are processed with AI to extract information
    - AI classifies document type and extracts key data
    - Can be processed asynchronously or synchronously
    - Can specify storage location (cloud, sharepoint, cch)
    - Can specify document category (client, internal, permanent)
    """
    if project_id not in projects_data:
        raise HTTPException(404, "Project not found")

    # Validate storage location
    valid_storage_locations = ["cloud", "sharepoint", "cch_document_storage", "keyveve_storage"]
    if storage_location not in valid_storage_locations:
        raise HTTPException(400, f"Invalid storage location. Must be one of: {', '.join(valid_storage_locations)}")

    # Validate document category
    valid_categories = ["client", "internal", "permanent"]
    if doc_category not in valid_categories:
        raise HTTPException(400, f"Invalid document category. Must be one of: {', '.join(valid_categories)}")

    # Save file to temp location
    file_ext = file.filename.split(".")[-1] if "." in file.filename else "pdf"
    stored_filename = f"{uuid.uuid4()}.{file_ext}"
    filepath = f"/tmp/{stored_filename}"

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Initial document record with placeholder values
    doc_id = f"doc-{uuid.uuid4()}"
    doc_type = "Processing..."
    extracted_data = "Document is being analyzed"

    # Create document record
    doc_record = {
        "doc_id": doc_id,
        "original_name": file.filename,
        "stored_name": stored_filename,
        "doc_type": doc_type,
        "extracted_data": extracted_data,
        "doc_category": doc_category,
        "storage_location": storage_location,
        "upload_timestamp": datetime.datetime.now().isoformat()
    }

    # Store initial doc record
    doc_store[doc_id] = {
        "file_name": stored_filename,
        "doc_type": doc_type,
        "extracted_data": extracted_data,
        "doc_category": doc_category,
        "storage_location": storage_location
    }

    # Add to project's doc list
    projects_data[project_id]["docs"].append(doc_record)

    # Update project timestamp
    projects_data[project_id]["updated_at"] = datetime.datetime.now().isoformat()

    # Process with AI in background
    if process_async:
        background_tasks.add_task(
            process_document_and_update_project,
            project_id,
            doc_id,
            filepath,
            file.filename
        )
    else:
        # For synchronous processing, do it now
        ai_result = await process_document_with_ai(filepath, file.filename)
        doc_type = ai_result.get("doc_type", "Unknown")
        extracted_data = ai_result.get("extracted_data", "No data extracted")
        ai_doc_category = ai_result.get("doc_category", doc_category)
        ai_storage_location = ai_result.get("suggested_storage", storage_location)

        # Update doc store
        doc_store[doc_id]["doc_type"] = doc_type
        doc_store[doc_id]["extracted_data"] = extracted_data
        doc_store[doc_id]["doc_category"] = ai_doc_category
        doc_store[doc_id]["storage_location"] = ai_storage_location

        # Update project doc
        for doc in projects_data[project_id]["docs"]:
            if doc["doc_id"] == doc_id:
                doc["doc_type"] = doc_type
                doc["extracted_data"] = extracted_data
                doc["doc_category"] = ai_doc_category
                doc["storage_location"] = ai_storage_location
                break

    # Create notification
    background_tasks.add_task(
        create_notification,
        project_id,
        "info",
        f"New document uploaded: {file.filename}"
    )

    return DocumentUploadResponse(
        doc_id=doc_id,
        doc_type=doc_type,
        extracted_data=extracted_data,
        original_name=file.filename,
        stored_name=stored_filename,
        storage_location=storage_location,
        doc_category=doc_category
    )


@app.patch("/documents/{doc_id}")
async def update_document(
        doc_id: str,
        update: DocumentUpdateRequest
):
    """
    Update document metadata.

    - Can update document type, storage location, category
    - Can associate document with tasks
    """
    # First check if document exists
    if doc_id not in doc_store:
        raise HTTPException(404, "Document not found")

    # Update document store
    if update.doc_type is not None:
        doc_store[doc_id]["doc_type"] = update.doc_type

    if update.storage_location is not None:
        valid_storage_locations = ["cloud", "sharepoint", "cch_document_storage", "keyveve_storage"]
        if update.storage_location not in valid_storage_locations:
            raise HTTPException(400, f"Invalid storage location. Must be one of: {', '.join(valid_storage_locations)}")
        doc_store[doc_id]["storage_location"] = update.storage_location

    if update.doc_category is not None:
        valid_categories = ["client", "internal", "permanent"]
        if update.doc_category not in valid_categories:
            raise HTTPException(400, f"Invalid document category. Must be one of: {', '.join(valid_categories)}")
        doc_store[doc_id]["doc_category"] = update.doc_category

    # Update document in all projects
    for project_id, project in projects_data.items():
        for i, doc in enumerate(project["docs"]):
            if doc["doc_id"] == doc_id:
                if update.doc_type is not None:
                    project["docs"][i]["doc_type"] = update.doc_type
                if update.storage_location is not None:
                    project["docs"][i]["storage_location"] = update.storage_location
                if update.doc_category is not None:
                    project["docs"][i]["doc_category"] = update.doc_category

                # Update project timestamp
                project["updated_at"] = datetime.datetime.now().isoformat()

                # If related tasks provided, update them to include this document
                if update.related_tasks:
                    for j, task in enumerate(project["tasks"]):
                        if isinstance(task, dict) and task.get("id") in update.related_tasks:
                            if "related_docs" not in project["tasks"][j]:
                                project["tasks"][j]["related_docs"] = []
                            if doc_id not in project["tasks"][j]["related_docs"]:
                                project["tasks"][j]["related_docs"].append(doc_id)

    return {
        "success": True,
        "doc_id": doc_id,
        "document": doc_store[doc_id]
    }


@app.get("/documents/{project_id}", response_model=List[Dict[str, Any]])
def get_project_documents(project_id: int):
    """Get all documents for a specific project."""
    if project_id not in projects_data:
        raise HTTPException(404, "Project not found")

    return projects_data[project_id]["docs"]


@app.post("/qa", response_model=Dict[str, str])
async def staff_qa(question_req: QuestionRequest):
    """
    AI-powered Q&A about project status, documents, etc.

    - Staff can ask questions in natural language
    - Questions can be about specific projects or general accounting knowledge
    - Can ask about missing documents, next steps, etc.
    """
    project_id = question_req.project_id
    question = question_req.question
    global_context = question_req.global_context

    # If project ID is provided, get project-specific context
    if project_id is not None and not global_context:
        if project_id not in projects_data:
            raise HTTPException(404, "Project not found")

        proj = projects_data[project_id]

        # Gather detailed context about the project
        status = proj["status"]
        project_created = proj.get("created_at", "unknown")
        days_in_current_status = "unknown"

        if "updated_at" in proj:
            try:
                updated = datetime.datetime.fromisoformat(proj["updated_at"])
                now = datetime.datetime.now()
                days_in_current_status = (now - updated).days
            except:
                days_in_current_status = "unknown"

        doc_summaries = [
            f"{d['doc_type']}: {d['extracted_data']} (uploaded: {d.get('upload_timestamp', 'unknown')}, stored in: {d.get('storage_location', 'cloud')}, category: {d.get('doc_category', 'client')})"
            for d in proj["docs"]
        ]

        # Get assigned staff names
        staff_names = []
        for staff_id in proj.get("assigned_staff", []):
            staff_role = proj.get("staff_roles", {}).get(staff_id, "staff")
            role_text = ""
            if staff_role == "point_of_contact":
                role_text = " (Point of Contact)"
            elif staff_role == "partner_assigned":
                role_text = " (Partner Assigned)"

            for staff in staff_members:
                if staff["id"] == staff_id:
                    staff_names.append(f"{staff['name']}{role_text} ({staff['role']})")
                    break

        staff_assignment = ", ".join(staff_names) if staff_names else "No staff assigned"

        # Get tasks with status
        task_list = []
        for task in proj.get("tasks", []):
            if isinstance(task, dict):
                task_status = task.get("status", "pending")
                task_assignees = []
                for staff_id in task.get("assigned_to", []):
                    for staff in staff_members:
                        if staff["id"] == staff_id:
                            task_assignees.append(staff["name"])
                            break
                assignee_str = f" (Assigned to: {', '.join(task_assignees)})" if task_assignees else ""
                deadline_str = f" (Due: {task.get('deadline')})" if task.get('deadline') else ""
                scheduled_str = ""
                if task.get("scheduled_start"):
                    start_date = datetime.datetime.fromisoformat(task["scheduled_start"]).strftime("%Y-%m-%d")
                    end_date = datetime.datetime.fromisoformat(task["scheduled_end"]).strftime("%Y-%m-%d") if task.get(
                        "scheduled_end") else "N/A"
                    scheduled_str = f" (Scheduled: {start_date} to {end_date})"

                task_list.append(
                    f"{task.get('title', 'Unknown task')} - Status: {task_status}{assignee_str}{deadline_str}{scheduled_str}")
            else:
                task_list.append(str(task))

        # Get recent messages
        msgs = [
            f"{m['sender']} ({m.get('timestamp', 'unknown')}): {m['text']}"
            for m in sorted(
                proj["messages"],
                key=lambda x: x.get("timestamp", ""),
                reverse=True
            )[:5]
        ]

        # Detailed context for the AI
        context = (
                f"Project #{project_id} for client '{proj['client_name']}'\n"
                f"Service Type: {proj.get('service_type', 'N/A')}\n"
                f"Current Status: {status} (for ~{days_in_current_status} days)\n"
                f"Created: {project_created}\n"
                f"Staff Assignment: {staff_assignment}\n\n"
                f"Documents ({len(proj['docs'])}):\n" + "\n".join([f"- {d}" for d in doc_summaries]) + "\n\n"
                                                                                                       f"Recent Messages:\n" + "\n".join(
            [f"- {m}" for m in msgs]) + "\n\n"
                                        f"Current Tasks:\n" + "\n".join([f"- {t}" for t in task_list])
        )
    else:
        # For global context, gather information about all projects
        projects_summary = []
        for pid, proj in projects_data.items():
            status = proj["status"]
            service = proj.get("service_type", "N/A")
            doc_count = len(proj.get("docs", []))

            # Get staff assignments
            staff_names = []
            for staff_id in proj.get("assigned_staff", []):
                staff_role = proj.get("staff_roles", {}).get(staff_id, "staff")
                role_text = ""
                if staff_role == "point_of_contact":
                    role_text = " (Point of Contact)"
                elif staff_role == "partner_assigned":
                    role_text = " (Partner Assigned)"

                for staff in staff_members:
                    if staff["id"] == staff_id:
                        staff_names.append(f"{staff['name']}{role_text}")
                        break

            staff_str = ", ".join(staff_names) if staff_names else "No staff assigned"

            # Project summary
            summary = (
                f"Project #{pid} - {proj['client_name']}\n"
                f"  Service: {service}\n"
                f"  Status: {status}\n"
                f"  Documents: {doc_count}\n"
                f"  Staff: {staff_str}\n"
            )
            projects_summary.append(summary)

        # Additional company-wide info
        total_projects = len(projects_data)
        active_projects = sum(1 for p in projects_data.values() if p["status"] != "Completed")

        # Get distribution by service type
        service_distribution = {}
        for p in projects_data.values():
            service = p.get("service_type", "N/A")
            service_distribution[service] = service_distribution.get(service, 0) + 1

        service_summary = "\n".join(
            [f"  {service}: {count} projects" for service, count in service_distribution.items()])

        # Create global context
        context = (
                f"Keyveve AI Accounting Firm - Global Overview\n\n"
                f"Total Projects: {total_projects}\n"
                f"Active Projects: {active_projects}\n"
                f"Completed Projects: {total_projects - active_projects}\n\n"
                f"Projects by Service Type:\n{service_summary}\n\n"
                f"Staff Members:\n" + "\n".join([f"  {s['name']} - {s['role']}" for s in staff_members]) + "\n\n"
                                                                                                           f"Project Summaries:\n" + "\n".join(
            projects_summary)
        )

    try:
        # Get Azure OpenAI model
        chat_model = await get_azure_openai_model()

        # Build messages for conversation
        messages = [
            SystemMessage(content=(
                "You are a knowledgeable accounting project assistant. Here's detailed context about the project or organization:\n\n"
                f"{context}\n\n"
                "Answer the user's question accurately and concisely based only on this context. "
                "If you don't know the answer, say so clearly rather than making assumptions. "
                "Focus on facts from the context, not general advice."
            )),
            HumanMessage(content=question)
        ]

        # Generate response
        response = await chat_model.agenerate([messages])
        answer = response.generations[0][0].text.strip()

        return {"answer": answer}

    except Exception as e:
        logger.error(f"Error in QA generation: {str(e)}")

        # Fallback response if AI fails
        if project_id is not None and not global_context:
            proj = projects_data[project_id]
            missing_docs = []
            if "W-2" not in [d.get("doc_type") for d in proj["docs"]]:
                missing_docs.append("W-2")
            if "ID" not in [d.get("doc_type") for d in proj["docs"]]:
                missing_docs.append("identification documents")

            if "missing documents" in question.lower():
                return {
                    "answer": f"For project #{project_id}, the missing documents are: {', '.join(missing_docs) if missing_docs else 'none'}"}
            elif "status" in question.lower():
                return {"answer": f"The project is currently in the '{status}' stage."}
            else:
                return {
                    "answer": f"The project has {len(proj['docs'])} documents uploaded and is in the '{status}' stage."}
        else:
            return {
                "answer": "I'm sorry, I couldn't process that question. Please try again with a more specific query."}


@app.post("/pricing/", response_model=PriceResponse)
def get_pricing_recommendation(price_req: PriceRequest):
    """
    Generate a pricing recommendation for a project.

    - Based on project details, document types, and complexity
    - Can include custom complexity factors for adjustments
    - Returns calculated price with explanation
    """
    if price_req.project_id not in projects_data:
        raise HTTPException(404, "Project not found")

    return calculate_price(price_req.project_id, price_req.complexity_factors)


@app.post("/integrations/engagement-letter", response_model=Dict[str, Any])
async def send_engagement_letter(
        project_id: int,
        background_tasks: BackgroundTasks
):
    """
    Send project data to engagement letter software (integration placeholder).

    - Updates project status to 'Awaiting Signature'
    - Simulates an integration with third-party engagement letter software
    - Creates a notification about the letter being sent
    """
    if project_id not in projects_data:
        raise HTTPException(404, "Project not found")

    # Update project status
    old_status = projects_data[project_id]["status"]
    projects_data[project_id]["status"] = "Awaiting Signature"
    projects_data[project_id]["updated_at"] = datetime.datetime.now().isoformat()

    # Update tasks
    service_type = projects_data[project_id].get("service_type", "Tax Return")
    new_tasks = generate_tasks_for_service_type(service_type, "Awaiting Signature")
    projects_data[project_id]["tasks"] = [task.dict() for task in new_tasks]

    # Create notification
    background_tasks.add_task(
        create_notification,
        project_id,
        "info",
        f"Engagement letter sent via {integrations['engagement_letter']['name']}"
    )

    # Simulate delay for "integration"
    await asyncio.sleep(1)

    return {
        "success": True,
        "project_id": project_id,
        "old_status": old_status,
        "new_status": "Awaiting Signature",
        "integration": integrations["engagement_letter"]["name"],
        "message": f"Project information sent to {integrations['engagement_letter']['name']} successfully"
    }


@app.post("/integrations/document-storage", response_model=Dict[str, Any])
async def send_to_document_storage(project_id: int, doc_ids: List[str], storage_location: str = "sharepoint"):
    """
    Send documents to long-term storage (integration placeholder).

    - Simulates sending documents to SharePoint, OneDrive, CCH, etc.
    - Can send specific documents or all project documents
    - Can specify target storage location
    """
    if project_id not in projects_data:
        raise HTTPException(404, "Project not found")

    # Validate storage location
    valid_storage_locations = ["cloud", "sharepoint", "cch_document_storage", "keyveve_storage"]
    if storage_location not in valid_storage_locations:
        raise HTTPException(400, f"Invalid storage location. Must be one of: {', '.join(valid_storage_locations)}")

    # Validate doc IDs
    project_doc_ids = [d["doc_id"] for d in projects_data[project_id]["docs"]]
    invalid_ids = [id for id in doc_ids if id not in project_doc_ids]

    if invalid_ids:
        raise HTTPException(400, f"Invalid document IDs: {', '.join(invalid_ids)}")

    # Get storage integration info
    storage_integration = integrations.get(storage_location, integrations["document_storage"])

    if not storage_integration["connected"]:
        return {
            "success": False,
            "message": f"{storage_integration['name']} integration is not connected",
            "project_id": project_id
        }

    # Update document storage locations
    for doc_id in doc_ids:
        # Update in doc store
        if doc_id in doc_store:
            doc_store[doc_id]["storage_location"] = storage_location

        # Update in project docs
        for doc in projects_data[project_id]["docs"]:
            if doc["doc_id"] == doc_id:
                doc["storage_location"] = storage_location
                break

    # Simulate delay for "integration"
    await asyncio.sleep(1)

    return {
        "success": True,
        "project_id": project_id,
        "doc_count": len(doc_ids),
        "storage_service": storage_integration["name"],
        "message": f"Documents sent to {storage_integration['name']} successfully"
    }

@app.patch("/documents/rename/{doc_id}")
def rename_document(doc_id: str, new_name: str = Form(...)):
    """
    Update a document's *final* file‑name (after review) and keep
    project‑embedded copies in sync.
    """
    if doc_id not in doc_store:
        raise HTTPException(404, "Not found")

    doc_store[doc_id]["final_name"] = new_name

    # propagate into every project that includes this doc
    for project in projects_data.values():
        for d in project["docs"]:
            if d["doc_id"] == doc_id:
                d["final_name"] = new_name
                break

    return {"success": True, "doc_id": doc_id, "final_name": new_name}


@app.post("/integrations/import-pms", response_model=Dict[str, Any])
async def import_from_practice_management(background_tasks: BackgroundTasks):
    """
    Import client data from practice management software (integration placeholder).

    - Simulates importing data from a practice management system
    - Creates projects for the imported clients
    """
    # Simulate PMS import by creating random clients
    client_names = [
        "Regional Hospital Group",
        "Peterson Family Trust",
        "First Street Bakery",
        "Sunshine Daycare LLC"
    ]

    service_types = [
        "Tax Return",
        "Bookkeeping",
        "Financial Planning",
        "Audit",
        "Advisory"
    ]

    imported_projects = []

    for _ in range(random.randint(1, 3)):
        client_name = random.choice(client_names)
        service = random.choice(service_types)

        # Create project
        proj_create = ProjectCreate(
            client_name=client_name,
            service_type=service,
            source="pms"
        )

        new_project = await create_project(proj_create, background_tasks)
        imported_projects.append({
            "id": new_project.id,
            "client_name": new_project.client_name,
            "service_type": new_project.service_type
        })

    return {
        "success": True,
        "message": f"Successfully imported {len(imported_projects)} clients from practice management software",
        "imported_count": len(imported_projects),
        "imported_projects": imported_projects
    }


@app.post("/integrations/import-csv", response_model=ImportCSVResponse)
async def import_from_csv(import_request: ImportCSVRequest, background_tasks: BackgroundTasks):
    """
    Import client data from CSV.

    - Process CSV content to create new projects
    - Expects CSV with headers including 'client_name' and 'service_type'
    """
    # Process the CSV content
    result = process_csv_import(import_request.file_content)

    # Create notifications for imported projects
    if result.success and result.imported_count > 0:
        for pid in result.project_ids:
            background_tasks.add_task(
                create_notification,
                pid,
                "info",
                f"Project imported from CSV"
            )

    return result


@app.get("/notifications/", response_model=List[Notification])
def get_notifications(
        project_id: Optional[int] = None,
        unread_only: bool = False,
        limit: int = Query(20, gt=0, le=100)
):
    """
    Get notifications, optionally filtered by project and read status.

    - Can filter to a specific project
    - Can filter to only unread notifications
    - Returns most recent notifications first
    """
    results = []

    for notif in notifications:
        # Apply filters
        if project_id is not None and notif["project_id"] != project_id:
            continue
        if unread_only and notif["read"]:
            continue

        results.append(Notification(**notif))

    # Sort by created_at, newest first
    results.sort(key=lambda x: x.created_at, reverse=True)

    return results[:limit]


@app.patch("/notifications/{notification_id}/read")
def mark_notification_read(notification_id: str):
    """Mark a notification as read."""
    for notif in notifications:
        if notif["id"] == notification_id:
            notif["read"] = True
            return {"success": True, "notification_id": notification_id}

    raise HTTPException(404, "Notification not found")


@app.get("/integrations/", response_model=Dict[str, Any])
def get_integrations():
    """Get status of all integrations."""
    return {
        "integrations": integrations,
        "count": len(integrations)
    }


@app.post("/integrations/connect", response_model=IntegrationResponse)
async def connect_integration(integration_req: IntegrationRequest):
    """
    Connect to or configure an integration.

    - Supports connecting, disconnecting, or syncing integrations
    - Placeholder for real integrations with practice management, document storage, etc.
    """
    if integration_req.integration_type not in integrations:
        raise HTTPException(400, f"Invalid integration type: {integration_req.integration_type}")

    integration = integrations[integration_req.integration_type]

    # Simulate integration action
    if integration_req.action == "connect":
        integration["connected"] = True
        integration["last_sync"] = datetime.datetime.now().isoformat()
        message = f"Connected to {integration['name']} successfully"
    elif integration_req.action == "disconnect":
        integration["connected"] = False
        message = f"Disconnected from {integration['name']}"
    elif integration_req.action == "sync":
        if not integration["connected"]:
            raise HTTPException(400, f"{integration['name']} is not connected")
        integration["last_sync"] = datetime.datetime.now().isoformat()
        message = f"Synced with {integration['name']} successfully"
    else:
        raise HTTPException(400, f"Invalid action: {integration_req.action}")

    # Simulate a delay for "integration" action
    await asyncio.sleep(1)

    return IntegrationResponse(
        integration_type=integration_req.integration_type,
        status="success",
        message=message,
        connected=integration["connected"]
    )


# New endpoint for document organization (mock implementation)
@app.post("/documents/organize/{project_id}")
async def organize_documents(
        project_id: int,
        background_tasks: BackgroundTasks
):
    """
    Simulate AI-powered document organization for a project.

    - Analyzes documents
    - Categorizes documents into folders based on type
    - Adds an AI‑style suggested_name if it is missing
    - Updates document metadata
    """
    if project_id not in projects_data:
        raise HTTPException(404, "Project not found")

    project = projects_data[project_id]
    service_type = project.get("service_type", "Tax Return")

    # Simulate AI processing delay
    await asyncio.sleep(2)

    # ---------------- folder assignment rules ---------------------------
    if service_type == "Tax Return":
        for doc in project["docs"]:
            name_l = doc["original_name"].lower()
            if "w2" in name_l or "w-2" in name_l or doc["doc_type"] == "W-2":
                doc["doc_type"] = "W-2"
                doc["doc_category"] = "client"
                doc["folder"] = "Income Documents"
            elif "1099" in name_l or "1099" in doc["doc_type"]:
                doc["doc_type"] = "1099"
                doc["doc_category"] = "client"
                doc["folder"] = "Income Documents"
            elif "1098" in name_l or "mortgage" in name_l:
                doc["doc_type"] = "1098 Mortgage Interest"
                doc["doc_category"] = "client"
                doc["folder"] = "Expense Documents"
            elif "donat" in name_l or "charit" in name_l:
                doc["doc_type"] = "Charitable Donations"
                doc["doc_category"] = "client"
                doc["folder"] = "Expense Documents"
            elif "prior" in name_l or "1040" in name_l:
                doc["doc_type"] = "Prior Year Return"
                doc["doc_category"] = "permanent"
                doc["folder"] = "Prior Year Returns"
            elif "workpaper" in name_l:
                doc["doc_type"] = "Tax Workpaper"
                doc["doc_category"] = "internal"
                doc["folder"] = "Workpapers"
            else:
                doc["folder"] = "Client Information"

    elif service_type == "Audit":
        for doc in project["docs"]:
            name_l = doc["original_name"].lower()
            if "balance" in name_l or "balance" in doc["doc_type"].lower():
                doc["doc_type"] = "Balance Sheet"
                doc["doc_category"] = "client"
                doc["folder"] = "Financial Statements"
            elif "income" in name_l or "income" in doc["doc_type"].lower():
                doc["doc_type"] = "Income Statement"
                doc["doc_category"] = "client"
                doc["folder"] = "Financial Statements"
            elif "cash flow" in name_l:
                doc["doc_type"] = "Cash Flow Statement"
                doc["doc_category"] = "client"
                doc["folder"] = "Financial Statements"
            elif "inventory" in name_l:
                doc["doc_type"] = "Inventory Listing"
                doc["doc_category"] = "client"
                doc["folder"] = "Audit Evidence"
            elif "receivable" in name_l or "aging" in name_l:
                doc["doc_type"] = "A/R Aging Schedule"
                doc["doc_category"] = "client"
                doc["folder"] = "Audit Evidence"
            elif "risk" in name_l:
                doc["doc_type"] = "Risk Assessment"
                doc["doc_category"] = "internal"
                doc["folder"] = "Audit Planning"
            elif "sampling" in name_l:
                doc["doc_type"] = "Sampling Methodology"
                doc["doc_category"] = "internal"
                doc["folder"] = "Audit Planning"
            elif "planning" in name_l:
                doc["doc_type"] = "Audit Planning Memo"
                doc["doc_category"] = "internal"
                doc["folder"] = "Audit Planning"
            elif "bank" in name_l or "statement" in name_l:
                doc["doc_type"] = "Bank Statement"
                doc["doc_category"] = "client"
                doc["folder"] = "Audit Evidence"
            else:
                doc["folder"] = "Client Information"

    elif service_type == "Bookkeeping":
        for doc in project["docs"]:
            name_l = doc["original_name"].lower()
            if "bank" in name_l and "statement" in name_l:
                doc["doc_type"] = "Bank Statement"
                doc["doc_category"] = "client"
                doc["folder"] = "Bank Statements"
            elif "credit" in name_l and "card" in name_l:
                doc["doc_type"] = "Credit Card Statement"
                doc["doc_category"] = "client"
                doc["folder"] = "Bank Statements"
            elif "invoice" in name_l or doc["doc_type"] == "Invoice":
                doc["doc_type"] = "Invoice"
                doc["doc_category"] = "client"
                doc["folder"] = "Invoices"
            elif "payable" in name_l:
                doc["doc_type"] = "Accounts Payable"
                doc["doc_category"] = "client"
                doc["folder"] = "Financial Reports"
            elif "payroll" in name_l:
                doc["doc_type"] = "Payroll Summary"
                doc["doc_category"] = "client"
                doc["folder"] = "Payroll"
            elif "reconciliation" in name_l:
                doc["doc_type"] = "Bank Reconciliation"
                doc["doc_category"] = "internal"
                doc["folder"] = "Workpapers"
            elif "depreciation" in name_l:
                doc["doc_type"] = "Depreciation Schedule"
                doc["doc_category"] = "internal"
                doc["folder"] = "Workpapers"
            elif "chart" in name_l and "account" in name_l:
                doc["doc_type"] = "Chart of Accounts"
                doc["doc_category"] = "permanent"
                doc["folder"] = "Permanent"
            elif "receipt" in name_l:
                doc["doc_type"] = "Receipt"
                doc["doc_category"] = "client"
                doc["folder"] = "Receipts"
            else:
                doc["folder"] = "Client Information"

    # ---------------- add suggested_name if missing ----------------------
    for doc in project["docs"]:
        if not doc.get("suggested_name"):
            slug = re.sub(r"\s+", "_", project["client_name"].lower())
            ext = os.path.splitext(doc["original_name"])[1] or ".pdf"
            doc["suggested_name"] = (
                f"{slug}_{datetime.datetime.now().year}_{doc['doc_type'].replace(' ','_').lower()}{ext}"
            )

    # ---------------- notification --------------------------------------
    notifications.append(
        {
            "id": str(uuid.uuid4()),
            "project_id": project_id,
            "type": "info",
            "message": f"AI successfully organized {len(project['docs'])} documents for project #{project_id}",
            "created_at": datetime.datetime.now().isoformat(),
            "read": False,
        }
    )

    return {
        "success": True,
        "project_id": project_id,
        "document_count": len(project["docs"]),
        "message": "Documents analyzed and organized successfully",
    }



# New endpoint for sending documents to storage
@app.post("/documents/send-to-storage/{project_id}")
async def send_documents_to_storage(
        project_id: int,
        storage_type: str = "keyveve_storage"
):
    """
    Simulate sending documents to selected storage system.

    - Can select from keyveve_storage, cch_document_storage, or sharepoint
    - Updates document metadata with new storage location
    """
    if project_id not in projects_data:
        raise HTTPException(404, "Project not found")

    # Validate storage type
    valid_storage_types = ["keyveve_storage", "cch_document_storage", "sharepoint", "cloud"]
    if storage_type not in valid_storage_types:
        raise HTTPException(400, f"Invalid storage type. Must be one of: {', '.join(valid_storage_types)}")

    # Get project
    project = projects_data[project_id]

    # Update storage location for all documents
    for doc in project["docs"]:
        doc["storage_location"] = storage_type

    # Simulate processing delay
    await asyncio.sleep(1.5)

    # Create notification
    storage_name = "Keyveve Storage" if storage_type == "keyveve_storage" else \
        "CCH Document Storage" if storage_type == "cch_document_storage" else \
            "SharePoint" if storage_type == "sharepoint" else "Cloud Storage"

    notification_id = str(uuid.uuid4())
    notification = {
        "id": notification_id,
        "project_id": project_id,
        "type": "info",
        "message": f"All documents successfully sent to {storage_name}",
        "created_at": datetime.datetime.now().isoformat(),
        "read": False
    }
    notifications.append(notification)

    return {
        "success": True,
        "project_id": project_id,
        "storage_type": storage_type,
        "storage_name": storage_name,
        "document_count": len(project["docs"]),
        "message": f"Documents successfully sent to {storage_name}"
    }


# ------------------------------
# Initialize dummy data and run
# ------------------------------
init_dummy_data()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
