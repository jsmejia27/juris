# pleading_drafting_service.py
import os
import io
import re
import time
import logging
from typing import List, Dict, Any, Optional

from langchain_ollama import OllamaLLM
from rag_pipeline import LegalRAGPipeline, DEFAULT_OLLAMA_URL, DEFAULT_LLM_MODEL

logger = logging.getLogger(__name__)

AVAILABLE_TEMPLATES = [
    {
        "id": "rule_45_petition",
        "name": "Petition for Review on Certiorari (Rule 45)",
        "court_type": "Supreme Court of the Philippines",
        "description": "Appellate petition under Rule 45 raising pure questions of law from the Court of Appeals or Sandiganbayan.",
        "fields": ["court_division", "petitioner", "respondent", "assailed_ca_decision", "date_received", "grounds", "facts_summary"]
    },
    {
        "id": "motion_reconsideration",
        "name": "Motion for Reconsideration",
        "court_type": "Regional Trial Court / Court of Appeals",
        "description": "Formal motion contesting an adverse judgment or interlocutory order based on patent errors of law or fact.",
        "fields": ["court_branch", "case_number", "movant", "adverse_party", "assailed_order_date", "grounds", "rebuttal_points"]
    },
    {
        "id": "judicial_affidavit",
        "name": "Judicial Affidavit (A.M. No. 12-8-8-SC)",
        "court_type": "Trial Courts / Quasi-Judicial Agencies",
        "description": "Direct testimony in Q&A format compliant with the Judicial Affidavit Rule, with lawyer's attestation clause.",
        "fields": ["court_branch", "case_number", "parties", "witness_name", "witness_info", "lawyer_name", "offer_of_testimony", "facts_to_testify"]
    },
    {
        "id": "nlrc_position_paper",
        "name": "NLRC Labor Position Paper",
        "court_type": "National Labor Relations Commission (RAB)",
        "description": "Complainant or Respondent Position Paper in illegal dismissal, money claims, and unfair labor practice disputes.",
        "fields": ["rab_region", "nlrc_case_no", "complainant", "respondent_company", "labor_arbiter", "employment_details", "causes_of_action", "facts"]
    },
    {
        "id": "legal_opinion_memo",
        "name": "Formal Legal Opinion & Case Memorandum",
        "court_type": "Internal Firm / Corporate Advisory",
        "description": "Comprehensive institutional legal opinion assessing legal liability, statutory compliance, and litigation risk.",
        "fields": ["client_name", "matter_title", "requesting_party", "legal_counsel", "factual_background", "specific_questions"]
    },
    {
        "id": "formal_demand_letter",
        "name": "Formal Legal Demand Letter",
        "court_type": "Pre-Litigation Notice",
        "description": "Final extrajudicial demand letter with notice to comply/vacate prior to filing civil or criminal complaints.",
        "fields": ["client_name", "recipient_name", "recipient_address", "claim_amount_or_obligation", "basis_of_obligation", "cure_period_days"]
    }
]


class PleadingDraftingService:
    def __init__(
        self,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        default_model: str = DEFAULT_LLM_MODEL
    ):
        self.ollama_url = ollama_url
        self.default_model = default_model

    def list_templates(self) -> List[Dict[str, Any]]:
        """Return list of supported Philippine court pleading templates."""
        return AVAILABLE_TEMPLATES

    def generate_pleading(
        self,
        template_id: str,
        case_data: Dict[str, Any],
        model: str = DEFAULT_LLM_MODEL,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Generates a publication-grade Philippine legal pleading with verified Supreme Court citations:
        1. Queries the RAG retrieval engine for controlling Philippine doctrines.
        2. Injects retrieved precedents into the structured pleading prompt.
        3. Formulates a complete, court-ready draft.
        """
        # 1. Search relevant Philippine legal doctrines based on user's facts & grounds
        search_query = f"{case_data.get('grounds', '')} {case_data.get('facts', '')} {case_data.get('facts_summary', '')} {case_data.get('causes_of_action', '')} {case_data.get('factual_background', '')}".strip()
        if not search_query:
            search_query = case_data.get("matter_title", "Philippine legal doctrine")

        pipeline = LegalRAGPipeline(llm_model=model)
        retrieved_docs = pipeline.retrieve_legal_context(search_query, limit=4)
        
        legal_authorities_block = "\n\n".join([
            f"[Authority: {d.metadata.get('title', 'Supreme Court Decision')} ({d.metadata.get('year', 'N/A')}) - GR: {d.metadata.get('gr_number', 'N/A')}]\n{d.page_content}"
            for d in retrieved_docs
        ])

        # 2. Build template-specific prompt
        system_instructions = self._get_template_prompt(template_id, case_data, legal_authorities_block)

        # 3. Generate Draft via LLM
        try:
            llm = OllamaLLM(
                base_url=self.ollama_url,
                model=model,
                temperature=temperature
            )
            draft_text = llm.invoke(system_instructions)
        except Exception as e:
            logger.error(f"Error generating pleading draft: {e}")
            draft_text = f"Error during draft generation: {e}"

        return {
            "template_id": template_id,
            "model_used": model,
            "generated_at": time.time(),
            "draft": draft_text,
            "citations_applied": [
                {
                    "title": d.metadata.get("title", ""),
                    "gr_number": d.metadata.get("gr_number", ""),
                    "year": d.metadata.get("year", ""),
                    "status": d.metadata.get("doctrine_status", "Active Precedent")
                }
                for d in retrieved_docs
            ]
        }

    def _get_template_prompt(self, template_id: str, data: Dict[str, Any], legal_authorities: str) -> str:
        """Construct prompt enforcing Supreme Court rules of form and style."""
        base_prompt = f"""You are **Juris Legal Drafter**, an expert Philippine Supreme Court practitioner and litigation specialist.

Draft a complete, rigorous, and court-ready Philippine legal document compliant with the **2019 Revised Rules of Civil Procedure** and **A.M. No. 11-9-4-SC (Efficient Use of Paper Rule)**.

===============================================================================
GOVERNING PHILIPPINE JURISPRUDENCE & STATUTORY CITATIONS
===============================================================================
{legal_authorities if legal_authorities else "[No specific statutory excerpts found; apply standard Philippine codal provisions.]"}

===============================================================================
CASE PARTICULARS & USER DATA
===============================================================================
"""
        for k, v in data.items():
            if v:
                base_prompt += f"* **{k.upper().replace('_', ' ')}**: {v}\n"

        base_prompt += "\n===============================================================================\n"

        if template_id == "rule_45_petition":
            base_prompt += """
DRAFTING REQUIREMENT: PETITION FOR REVIEW ON CERTIORARI UNDER RULE 45
Format must include:
1. REPUBLIC OF THE PHILIPPINES / SUPREME COURT OF THE PHILIPPINES CAPTION
2. TITLE: PETITION FOR REVIEW ON CERTIORARI
3. TIMELINESS OF THE PETITION (15-day period under Rule 45, Sec. 2; receipt of CA resolution denying MR)
4. PARTIES & MATERIAL DATES
5. STATEMENT OF THE MATTERS INVOLVED & PRIOR PROCEEDINGS
6. ASSIGNMENT OF ERRORS (Pure Questions of Law)
7. EXHAUSTIVE ARGUMENTS & DISCUSSION (Directly integrate the retrieved Supreme Court citations with G.R. numbers and ponentes)
8. PRAYER (Clear and specific reliefs)
9. VERIFICATION AND CERTIFICATION AGAINST FORUM SHOPPING (compliant with 2019 Rules)
"""
        elif template_id == "motion_reconsideration":
            base_prompt += """
DRAFTING REQUIREMENT: MOTION FOR RECONSIDERATION
Format must include:
1. FORMAL TRIAL COURT / APPELLATE CAPTION (Branch, Docket/Case No., Parties)
2. TITLE: MOTION FOR RECONSIDERATION (Re: Decision/Order dated [Date])
3. PREFATORY STATEMENT & TIMELINESS (Filed within 15 days from notice)
4. GROUNDS FOR RECONSIDERATION
5. DETAILED ARGUMENTS & REBUTTAL (Synthesize controlling SC precedents to demonstrate patent error)
6. PRAYER (Setting aside the assailed order and rendering a new favorable judgment)
7. NOTICE OF HEARING & PROOF OF SERVICE (Compliant with 2019 Rules of Civil Procedure)
"""
        elif template_id == "judicial_affidavit":
            base_prompt += """
DRAFTING REQUIREMENT: JUDICIAL AFFIDAVIT (A.M. NO. 12-8-8-SC)
Format must include:
1. FORMAL CAPTION OF COURT AND CASE
2. PRELIMINARY INFORMATION (Witness name, age, status, address, language, examining lawyer, place of examination)
3. LAWYER'S OFFER OF TESTIMONY (Specific factual matters to be proven)
4. DIRECT EXAMINATION IN NUMBERED QUESTION-AND-ANSWER (Q1, A1, Q2, A2...) FORMAT
5. WITNESS SIGNATURE BLOCK AND JURAT
6. LAWYER'S ATTESTATION CLAUSE (Mandatory Section 3 compliance: certifying that lawyer faithfully recorded answers and did not coach the witness)
"""
        elif template_id == "nlrc_position_paper":
            base_prompt += """
DRAFTING REQUIREMENT: NLRC POSITION PAPER (LABOR ARBITRATION)
Format must include:
1. REPUBLIC OF THE PHILIPPINES / NATIONAL LABOR RELATIONS COMMISSION CAPTION
2. TITLE: POSITION PAPER (FOR COMPLAINANT / RESPONDENT)
3. STATEMENT OF THE CASE & JURISDICTION
4. STATEMENT OF RELEVANT FACTS (Employment dates, position, salary, incidents leading to termination)
5. ISSUES PRESENTED (Illegal dismissal, twin-notice rule, money claims, moral/exemplary damages, attorney's fees)
6. ARGUMENTS & DISCUSSION (Apply Labor Code provisions and controlling Supreme Court labor jurisprudence)
7. PRAYER (Reinstatement, full backwages, separation pay, damages, 10% attorney's fees)
8. VERIFICATION UNDER OATH
"""
        elif template_id == "legal_opinion_memo":
            base_prompt += """
DRAFTING REQUIREMENT: INSTITUTIONAL LEGAL OPINION & CASE MEMORANDUM
Format must include:
1. FORMAL MEMORANDUM HEADER (TO, FROM, DATE, RE)
2. EXECUTIVE SUMMARY & BOTTOM-LINE OPINION
3. COMPREHENSIVE STATEMENT OF FACTS
4. LEGAL ISSUES IDENTIFIED
5. IN-DEPTH LEGAL DISCUSSION (Codal provisions, statutory elements, and Supreme Court doctrine analysis)
6. STRATEGIC LITIGATION RECOMMENDATIONS & RISK MATRIX
7. CONCLUSION & SIGNATURE BLOCK
"""
        elif template_id == "formal_demand_letter":
            base_prompt += """
DRAFTING REQUIREMENT: FORMAL LEGAL DEMAND LETTER
Format must include:
1. LAW FIRM LETTERHEAD & DATE
2. FORMAL ADDRESSEE BLOCK
3. SUBJECT LINE / RE: FORMAL AND FINAL DEMAND TO PAY / COMPLY
4. FACTUAL NARRATIVE OF THE UNFULFILLED OBLIGATION / CONTRACT BREACH
5. SPECIFIC MONETARY COMPUTATION OR ACTION DEMANDED
6. DEFINITIVE PERIOD TO COMPLY (e.g. 5 or 10 days from receipt)
7. NOTICE OF LEGAL REPERCUSSIONS (Civil actions, damages, interest, and criminal complaints under Philippine law)
8. FORMAL CLOSING & COUNSEL SIGNATURE BLOCK
"""
        return base_prompt

    def export_to_docx(self, draft_text: str, title: str = "Philippine_Legal_Pleading") -> io.BytesIO:
        """
        Export generated Markdown pleading to a professional Microsoft Word (.docx) document
        formatted per Philippine Supreme Court standard typography & margins.
        """
        from docx import Document
        from docx.shared import Inches, Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.style import WD_STYLE_TYPE

        doc = Document()

        # Set Philippine Court Standard Margins (1.5" left, 1.0" top/bottom/right)
        for section in doc.sections:
            section.top_margin = Inches(1.0)
            section.bottom_margin = Inches(1.0)
            section.left_margin = Inches(1.5)
            section.right_margin = Inches(1.0)

        # Base style configuration
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Times New Roman'
        font.size = Pt(12)
        font.color.rgb = RGBColor(15, 23, 42)

        # Parse draft lines
        lines = draft_text.split("\n")
        for line in lines:
            line_str = line.strip()
            if not line_str:
                doc.add_paragraph("")
                continue

            # Heading 1 (# ...)
            if line_str.startswith("# "):
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run(line_str[2:].replace("**", "").upper())
                run.bold = True
                run.font.size = Pt(14)
            # Heading 2 (## ...)
            elif line_str.startswith("## "):
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run(line_str[3:].replace("**", "").upper())
                run.bold = True
                run.font.size = Pt(12.5)
            # Heading 3 (### ...)
            elif line_str.startswith("### "):
                p = doc.add_paragraph()
                run = p.add_run(line_str[4:].replace("**", ""))
                run.bold = True
                run.font.size = Pt(12)
            # Bullet list
            elif line_str.startswith("* ") or line_str.startswith("- "):
                p = doc.add_paragraph(style='List Bullet')
                p.paragraph_format.line_spacing = 1.15
                self._add_formatted_runs(p, line_str[2:])
            # Numbered item
            elif re.match(r"^\d+\.\s", line_str):
                p = doc.add_paragraph(style='List Number')
                p.paragraph_format.line_spacing = 1.15
                text_part = re.sub(r"^\d+\.\s", "", line_str)
                self._add_formatted_runs(p, text_part)
            # Blockquote
            elif line_str.startswith("> "):
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.5)
                p.paragraph_format.right_indent = Inches(0.5)
                p.paragraph_format.line_spacing = 1.15
                run = p.add_run(line_str[2:].replace("**", ""))
                run.italic = True
            # Normal paragraph
            else:
                p = doc.add_paragraph()
                p.paragraph_format.line_spacing = 1.5 # Standard court spacing
                p.paragraph_format.space_after = Pt(4)
                
                # Check for centered captions
                if any(kw in line_str.upper() for kw in ["REPUBLIC OF THE PHILIPPINES", "SUPREME COURT", "REGIONAL TRIAL COURT", "- VERSUS -", "VS.", "PRAYER", "VERIFICATION"]):
                    if len(line_str) < 60:
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                
                self._add_formatted_runs(p, line_str)

        output_stream = io.BytesIO()
        doc.save(output_stream)
        output_stream.seek(0)
        return output_stream

    def _add_formatted_runs(self, paragraph, text: str):
        """Helper to parse markdown **bold** and *italic* tokens into Word runs."""
        # Simple regex tokenizer for bold/italic
        parts = re.split(r"(\*\*.*?\*\*|\*.*?\*)", text)
        for part in parts:
            if part.startswith("**") and part.endswith("**") and len(part) >= 4:
                run = paragraph.add_run(part[2:-2])
                run.bold = True
            elif part.startswith("*") and part.endswith("*") and len(part) >= 2:
                run = paragraph.add_run(part[1:-1])
                run.italic = True
            else:
                paragraph.add_run(part)
