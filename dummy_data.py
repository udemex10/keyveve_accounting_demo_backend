# dummy_data.py
# ---------------------------------------------------------------------------
#  Full demo data set for Keyveve prototype
# ---------------------------------------------------------------------------
import datetime
import uuid
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
#  small helpers
# ---------------------------------------------------------------------------
def _iso(dt: datetime.datetime) -> str:
    """Return ISO‑8601 string."""
    return dt.isoformat()


def _doc(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure every document carries the extra prototype fields.
    """
    rec.setdefault("status", "awaiting_review")    # workflow: awaiting_review ▸ reviewed ▸ signed ▸ filed
    rec.setdefault("suggested_name", None)         # to be filled by AI
    rec.setdefault("final_name", None)             # user‑approved
    return rec


# ---------------------------------------------------------------------------
#  PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------
def init_dummy_data(
    *,
    projects_data: Dict[int, Dict[str, Any]],
    doc_store: Dict[str, Dict[str, Any]],
    staff_members: List[Dict[str, Any]],
    notifications: List[Dict[str, Any]],
    generate_tasks_for_service_type,
) -> int:
    """
    Populate the in‑memory stores with the original rich sample data
    (now including the new document fields) and return the next project ID.
    """

    # -----------------------------------------------------------------------
    #  Staff (unchanged)
    # -----------------------------------------------------------------------
    staff_members[:] = [
        {
            "id": "staff-001",
            "name": "Sarah Johnson",
            "email": "sarah.johnson@keyveve.com",
            "role": "Senior Accountant",
            "avatar_url": "https://randomuser.me/api/portraits/women/20.jpg",
        },
        {
            "id": "staff-002",
            "name": "Michael Chen",
            "email": "michael.chen@keyveve.com",
            "role": "Tax Specialist",
            "avatar_url": "https://randomuser.me/api/portraits/men/32.jpg",
        },
        {
            "id": "staff-003",
            "name": "Jessica Rodriguez",
            "email": "jessica.rodriguez@keyveve.com",
            "role": "Audit Manager",
            "avatar_url": "https://randomuser.me/api/portraits/women/43.jpg",
        },
        {
            "id": "staff-004",
            "name": "David Kim",
            "email": "david.kim@keyveve.com",
            "role": "Bookkeeper",
            "avatar_url": "https://randomuser.me/api/portraits/men/55.jpg",
        },
    ]

    # -----------------------------------------------------------------------
    #  Common time helpers
    # -----------------------------------------------------------------------
    now = datetime.datetime.now()
    td = datetime.timedelta

    # -----------------------------------------------------------------------
    #  PROJECT #1  – John Doe  (Docs Requested, Tax Return)
    # -----------------------------------------------------------------------
    p1_tasks = generate_tasks_for_service_type("Tax Return", "Docs Requested")

    projects_data[1] = {
        "id": 1,
        "client_name": "John Doe",
        "service_type": "Tax Return",
        "status": "Docs Requested",
        "docs": [
            _doc(
                {
                    "doc_id": "doc-abc1",
                    "original_name": "prior_year_1040.pdf",
                    "stored_name": "abc1.pdf",
                    "doc_type": "1040 Return",
                    "doc_category": "permanent",
                    "storage_location": "cch_document_storage",
                    "extracted_data": "Prior‑year 1040 showing AGI $75,000",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-abc2",
                    "original_name": "john_doe_w2_2023.pdf",
                    "stored_name": "abc2.pdf",
                    "doc_type": "W-2",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "W‑2 wages $82,500 / federal tax $16,500",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-abc3",
                    "original_name": "john_doe_1099-int.pdf",
                    "stored_name": "abc3.pdf",
                    "doc_type": "1099‑INT",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "Interest income $1,250 (First National Bank)",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-abc4",
                    "original_name": "john_doe_mortgage_interest.pdf",
                    "stored_name": "abc4.pdf",
                    "doc_type": "1098",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "Mortgage interest paid $12,400",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-abc5",
                    "original_name": "john_doe_charitable_donations.pdf",
                    "stored_name": "abc5.pdf",
                    "doc_type": "Donation Receipt",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "Charitable donations total $5,600",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-abc6",
                    "original_name": "tax_planning_workpaper.xlsx",
                    "stored_name": "abc6.xlsx",
                    "doc_type": "Workpaper",
                    "doc_category": "internal",
                    "storage_location": "sharepoint",
                    "extracted_data": "Internal tax‑planning scenarios",
                }
            ),
        ],
        "tasks": [t.dict() for t in p1_tasks],
        "messages": [
            {
                "id": "msg-111",
                "project_id": 1,
                "sender": "staff",
                "sender_id": "staff-002",
                "text": "Hello John, please upload your W‑2 for this year.",
                "timestamp": _iso(now - td(days=2)),
            }
        ],
        "assigned_staff": ["staff-002"],
        "staff_roles": {"staff-002": "point_of_contact"},
        "created_at": _iso(now - td(days=5)),
        "updated_at": _iso(now - td(days=2)),
    }

    # -----------------------------------------------------------------------
    #  PROJECT #2  – Acme Corp  (Docs Received)
    # -----------------------------------------------------------------------
    p2_tasks = generate_tasks_for_service_type("Tax Return", "Docs Received")
    p2_tasks[0].related_docs = ["doc-xyz2", "doc-xyz3"]
    p2_tasks[1].related_docs = ["doc-xyz2", "doc-xyz3"]

    projects_data[2] = {
        "id": 2,
        "client_name": "Acme Corp",
        "service_type": "Tax Return",
        "status": "Docs Received",
        "docs": [
            _doc(
                {
                    "doc_id": "doc-xyz2",
                    "original_name": "acme_2023_invoice.pdf",
                    "stored_name": "xyz2.pdf",
                    "doc_type": "Invoice",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "Invoice #2023, vendor XYZ $12,500",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-xyz3",
                    "original_name": "acme_w-9.pdf",
                    "stored_name": "xyz3.pdf",
                    "doc_type": "W-9",
                    "doc_category": "permanent",
                    "storage_location": "cch_document_storage",
                    "extracted_data": "W‑9 EIN 12‑3456789",
                }
            ),
        ],
        "tasks": [t.dict() for t in p2_tasks],
        "messages": [
            {
                "id": "msg-222",
                "project_id": 2,
                "sender": "client",
                "text": "We have uploaded the invoices and W‑9 form. Please let us know next steps.",
                "timestamp": _iso(now - td(days=1)),
            },
            {
                "id": "msg-223",
                "project_id": 2,
                "sender": "staff",
                "sender_id": "staff-002",
                "text": "Thanks Acme, we will review and update you soon.",
                "timestamp": _iso(now),
            },
        ],
        "assigned_staff": ["staff-002", "staff-001"],
        "staff_roles": {"staff-002": "point_of_contact", "staff-001": "partner_assigned"},
        "created_at": _iso(now - td(days=7)),
        "updated_at": _iso(now),
    }

    # -----------------------------------------------------------------------
    #  PROJECT #3  – Jane Smith  (Awaiting Signature, Audit)
    # -----------------------------------------------------------------------
    p3_tasks = generate_tasks_for_service_type("Audit", "Awaiting Signature")

    projects_data[3] = {
        "id": 3,
        "client_name": "Jane Smith",
        "service_type": "Audit",
        "status": "Awaiting Signature",
        "docs": [
            _doc(
                {
                    "doc_id": "doc-jane1",
                    "original_name": "bank_statement_jan.pdf",
                    "stored_name": "jane1.pdf",
                    "doc_type": "Bank Statement",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "January balance $45,200",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-jane2",
                    "original_name": "balance_sheet_q4.pdf",
                    "stored_name": "jane2.pdf",
                    "doc_type": "Balance Sheet",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "Assets $230,500 (Q4)",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-jane3",
                    "original_name": "audit_planning_memo.pdf",
                    "stored_name": "jane3.pdf",
                    "doc_type": "Audit Planning Memo",
                    "doc_category": "internal",
                    "storage_location": "sharepoint",
                    "extracted_data": "Internal planning document",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-jane4",
                    "original_name": "income_statement_q4.pdf",
                    "stored_name": "jane4.pdf",
                    "doc_type": "Income Statement",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "Revenue $580k / Net income $145k (Q4)",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-jane5",
                    "original_name": "cash_flow_statement.pdf",
                    "stored_name": "jane5.pdf",
                    "doc_type": "Cash Flow Statement",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "Operating cash $210k (annual)",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-jane6",
                    "original_name": "inventory_listing.xlsx",
                    "stored_name": "jane6.xlsx",
                    "doc_type": "Inventory Report",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "1,250 items, value $320k",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-jane7",
                    "original_name": "accounts_receivable_aging.xlsx",
                    "stored_name": "jane7.xlsx",
                    "doc_type": "Accounts Receivable Aging",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "Receivables $125k / 90+ $15k",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-jane8",
                    "original_name": "audit_risk_assessment.docx",
                    "stored_name": "jane8.docx",
                    "doc_type": "Audit Risk Assessment",
                    "doc_category": "internal",
                    "storage_location": "sharepoint",
                    "extracted_data": "Key risk areas documented",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-jane9",
                    "original_name": "audit_sampling_workpaper.xlsx",
                    "stored_name": "jane9.xlsx",
                    "doc_type": "Audit Sampling Workpaper",
                    "doc_category": "internal",
                    "storage_location": "sharepoint",
                    "extracted_data": "Sampling methodology & selections",
                }
            ),
        ],
        "tasks": [t.dict() for t in p3_tasks],
        "messages": [
            {
                "id": "msg-333",
                "project_id": 3,
                "sender": "staff",
                "sender_id": "staff-003",
                "text": "Jane, your documents look good. We'll send an engagement letter shortly.",
                "timestamp": _iso(now - td(days=2)),
            },
            {
                "id": "msg-334",
                "project_id": 3,
                "sender": "staff",
                "sender_id": "staff-003",
                "text": "Engagement letter sent. Please sign at your earliest convenience.",
                "timestamp": _iso(now - td(hours=4)),
            },
        ],
        "assigned_staff": ["staff-003"],
        "staff_roles": {"staff-003": "partner_assigned"},
        "created_at": _iso(now - td(days=10)),
        "updated_at": _iso(now - td(hours=4)),
    }

    # -----------------------------------------------------------------------
    #  PROJECT #4  – Tech Startup Inc  (Project Started, Bookkeeping)
    # -----------------------------------------------------------------------
    p4_tasks = generate_tasks_for_service_type("Bookkeeping", "Project Started")

    projects_data[4] = {
        "id": 4,
        "client_name": "Tech Startup Inc",
        "service_type": "Bookkeeping",
        "status": "Project Started",
        "docs": [
            _doc(
                {
                    "doc_id": "doc-tech1",
                    "original_name": "q1_expenses.pdf",
                    "stored_name": "tech1.pdf",
                    "doc_type": "Expense Report",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "Q1 expenses $128,750",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-tech2",
                    "original_name": "q1_revenue.pdf",
                    "stored_name": "tech2.pdf",
                    "doc_type": "Revenue Report",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "Q1 revenue $215,300",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-tech3",
                    "original_name": "chart_of_accounts.xlsx",
                    "stored_name": "tech3.xlsx",
                    "doc_type": "Chart of Accounts",
                    "doc_category": "permanent",
                    "storage_location": "sharepoint",
                    "extracted_data": "Company chart of accounts",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-tech4",
                    "original_name": "bookkeeping_workpaper.xlsx",
                    "stored_name": "tech4.xlsx",
                    "doc_type": "Workpaper",
                    "doc_category": "internal",
                    "storage_location": "sharepoint",
                    "extracted_data": "Internal reconciliation workpaper",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-tech5",
                    "original_name": "q1_bank_statement_chase.pdf",
                    "stored_name": "tech5.pdf",
                    "doc_type": "Bank Statement",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "Q1 checking balance $45,230",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-tech6",
                    "original_name": "q1_bank_statement_wells.pdf",
                    "stored_name": "tech6.pdf",
                    "doc_type": "Bank Statement",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "Q1 savings balance $120,500",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-tech7",
                    "original_name": "q1_credit_card_statement.pdf",
                    "stored_name": "tech7.pdf",
                    "doc_type": "Credit Card Statement",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "Q1 credit‑card spend $28,450",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-tech8",
                    "original_name": "accounts_payable_aging.xlsx",
                    "stored_name": "tech8.xlsx",
                    "doc_type": "Accounts Payable Aging",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "Payables $62,300 / 60+ $8,500",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-tech9",
                    "original_name": "payroll_summary_q1.pdf",
                    "stored_name": "tech9.pdf",
                    "doc_type": "Payroll Summary",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "Q1 wages $235k / 15 employees",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-tech10",
                    "original_name": "bank_reconciliation.xlsx",
                    "stored_name": "tech10.xlsx",
                    "doc_type": "Bank Reconciliation",
                    "doc_category": "internal",
                    "storage_location": "sharepoint",
                    "extracted_data": "Reconciliation for checking account",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-tech11",
                    "original_name": "depreciation_schedule.xlsx",
                    "stored_name": "tech11.xlsx",
                    "doc_type": "Depreciation Schedule",
                    "doc_category": "internal",
                    "storage_location": "sharepoint",
                    "extracted_data": "Fixed‑asset depreciation schedule",
                }
            ),
        ],
        "tasks": [t.dict() for t in p4_tasks],
        "messages": [
            {
                "id": "msg-444",
                "project_id": 4,
                "sender": "client",
                "text": "Looking forward to working with your team on our bookkeeping needs.",
                "timestamp": _iso(now - td(days=5)),
            },
            {
                "id": "msg-445",
                "project_id": 4,
                "sender": "staff",
                "sender_id": "staff-004",
                "text": "We've set up your account in our system and will begin organizing your financials this week.",
                "timestamp": _iso(now - td(days=3)),
            },
        ],
        "assigned_staff": ["staff-004", "staff-001"],
        "staff_roles": {"staff-004": "point_of_contact", "staff-001": "partner_assigned"},
        "created_at": _iso(now - td(days=15)),
        "updated_at": _iso(now - td(days=3)),
    }

    # -----------------------------------------------------------------------
    #  PROJECT #5  – Smith Family LLC  (Financial Planning, Project Started)
    # -----------------------------------------------------------------------
    p5_tasks = generate_tasks_for_service_type("Financial Planning", "Project Started")

    projects_data[5] = {
        "id": 5,
        "client_name": "Smith Family LLC",
        "service_type": "Financial Planning",
        "status": "Project Started",
        "docs": [
            _doc(
                {
                    "doc_id": "doc-smith1",
                    "original_name": "investment_portfolio.pdf",
                    "stored_name": "smith1.pdf",
                    "doc_type": "Investment Portfolio",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "Portfolio value $1.2 M",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-smith2",
                    "original_name": "retirement_accounts.pdf",
                    "stored_name": "smith2.pdf",
                    "doc_type": "Retirement Accounts",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "401k, IRA totals listed",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-smith3",
                    "original_name": "financial_plan_draft.docx",
                    "stored_name": "smith3.docx",
                    "doc_type": "Financial Plan Draft",
                    "doc_category": "internal",
                    "storage_location": "sharepoint",
                    "extracted_data": "Draft plan & recommendations",
                }
            ),
        ],
        "tasks": [t.dict() for t in p5_tasks],
        "messages": [
            {
                "id": "msg-556",
                "project_id": 5,
                "sender": "client",
                "text": "We'd like to focus on retirement planning and college savings for our children.",
                "timestamp": _iso(now - td(days=7)),
            },
            {
                "id": "msg-557",
                "project_id": 5,
                "sender": "staff",
                "sender_id": "staff-001",
                "text": "Thanks for sharing your priorities— we'll incorporate them into your plan.",
                "timestamp": _iso(now - td(days=6)),
            },
        ],
        "assigned_staff": ["staff-001"],
        "staff_roles": {"staff-001": "partner_assigned"},
        "created_at": _iso(now - td(days=12)),
        "updated_at": _iso(now - td(days=3)),
    }

    # -----------------------------------------------------------------------
    #  PROJECT #6  – Metro Hospital  (Advisory, Pricing/Analysis)
    # -----------------------------------------------------------------------
    p6_tasks = generate_tasks_for_service_type("Advisory", "Pricing/Analysis")

    projects_data[6] = {
        "id": 6,
        "client_name": "Metro Hospital",
        "service_type": "Advisory",
        "status": "Pricing/Analysis",
        "docs": [
            _doc(
                {
                    "doc_id": "doc-metro1",
                    "original_name": "financial_statements_2023.pdf",
                    "stored_name": "metro1.pdf",
                    "doc_type": "Financial Statements",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "Annual revenue $45 M",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-metro2",
                    "original_name": "operational_metrics.xlsx",
                    "stored_name": "metro2.xlsx",
                    "doc_type": "Operational Metrics",
                    "doc_category": "client",
                    "storage_location": "cloud",
                    "extracted_data": "Key hospital KPIs",
                }
            ),
            _doc(
                {
                    "doc_id": "doc-metro3",
                    "original_name": "advisory_proposal_draft.docx",
                    "stored_name": "metro3.docx",
                    "doc_type": "Advisory Proposal",
                    "doc_category": "internal",
                    "storage_location": "sharepoint",
                    "extracted_data": "Draft proposal on efficiency consulting",
                }
            ),
        ],
        "tasks": [t.dict() for t in p6_tasks],
        "messages": [
            {
                "id": "msg-668",
                "project_id": 6,
                "sender": "client",
                "text": "We're interested in improving billing efficiency and reducing costs.",
                "timestamp": _iso(now - td(days=4)),
            },
            {
                "id": "msg-669",
                "project_id": 6,
                "sender": "staff",
                "sender_id": "staff-003",
                "text": "Reviewing your financials now; we'll prepare a customized proposal.",
                "timestamp": _iso(now - td(days=3)),
            },
        ],
        "assigned_staff": ["staff-003", "staff-001"],
        "staff_roles": {"staff-003": "point_of_contact", "staff-001": "partner_assigned"},
        "created_at": _iso(now - td(days=8)),
        "updated_at": _iso(now - td(days=1)),
    }

    # -----------------------------------------------------------------------
    #  push every document into the global doc_store
    # -----------------------------------------------------------------------
    for proj in projects_data.values():
        for d in proj["docs"]:
            doc_store[d["doc_id"]] = d.copy()

    # -----------------------------------------------------------------------
    #  notifications (four, unchanged except new ids)
    # -----------------------------------------------------------------------
    notifications[:] = [
        {
            "id": str(uuid.uuid4()),
            "project_id": 1,
            "type": "reminder",
            "message": "Client has not uploaded W‑2 within 3 days of request",
            "created_at": _iso(now),
            "read": False,
        },
        {
            "id": str(uuid.uuid4()),
            "project_id": 3,
            "type": "info",
            "message": "Engagement letter sent and awaiting signature",
            "created_at": _iso(now - td(hours=4)),
            "read": True,
        },
        {
            "id": str(uuid.uuid4()),
            "project_id": 5,
            "type": "info",
            "message": "New document uploaded by client: Investment Portfolio",
            "created_at": _iso(now - td(hours=8)),
            "read": False,
        },
        {
            "id": str(uuid.uuid4()),
            "project_id": 6,
            "type": "alert",
            "message": "Advisory proposal approval needed before proceeding",
            "created_at": _iso(now - td(hours=2)),
            "read": False,
        },
    ]

    # -----------------------------------------------------------------------
    return max(projects_data.keys()) + 1
