import os
from jinja2 import Environment, FileSystemLoader
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from utils.document_packet import DocumentPacket
from utils.logger import get_logger

logger = get_logger("report_generator")

class ReportGenerator:
    def __init__(self):
        self.app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.template_dir = os.path.join(self.app_dir, "reports", "templates")
        os.makedirs(self.template_dir, exist_ok=True)
        self._create_default_templates()

    def _create_default_templates(self):
        """Creates basic HTML templates if they are missing."""
        html_template_path = os.path.join(self.template_dir, "audit_report.html")
        if not os.path.exists(html_template_path):
            content = """<!DOCTYPE html>
<html>
<head>
    <title>DocuScan India - Audit Report</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background-color: #f8f9fa; color: #212529; }
        .container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .header { border-bottom: 2px solid #0056b3; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { margin: 0; color: #0056b3; }
        .status-badge { padding: 8px 16px; border-radius: 20px; font-weight: bold; color: white; display: inline-block; }
        .status-pass { background-color: #28a745; }
        .status-warn { background-color: #ffc107; color: #212529; }
        .status-fail { background-color: #dc3545; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 30px; }
        th, td { padding: 12px; border: 1px solid #dee2e6; text-align: left; }
        th { background-color: #f1f3f5; }
        .section-title { font-size: 20px; color: #343a40; margin-bottom: 15px; border-left: 4px solid #0056b3; padding-left: 10px; }
        .risk-score { font-size: 32px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>DocuScan India</h1>
                <p>Intelligent OCR & Verification System</p>
            </div>
            <div>
                <span class="status-badge status-{{ verdict.lower() }}">{{ verdict }}</span>
            </div>
        </div>

        <div class="section-title">Document Summary</div>
        <table>
            <tr>
                <th>Document Type</th>
                <td>{{ document_type }}</td>
                <th>Fraud Risk Score</th>
                <td class="risk-score" style="color: {% if risk_score < 25 %}#28a745{% elif risk_score < 50 %}#ffc107{% else %}#dc3545{% endif %}">{{ risk_score }}/100</td>
            </tr>
            <tr>
                <th>Source File</th>
                <td>{{ image_path }}</td>
                <th>Timestamp</th>
                <td>{{ timestamp }}</td>
            </tr>
        </table>

        <div class="section-title">Extracted Fields</div>
        <table>
            <thead>
                <tr>
                    <th>Field Name</th>
                    <th>Extracted Value</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
                {% for k, v in fields.items() %}
                <tr>
                    <td><strong>{{ k | replace('_', ' ') | title }}</strong></td>
                    <td>{{ v.value }}</td>
                    <td>{{ (v.confidence * 100) | round(1) }}%</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <div class="section-title">Validation Checks</div>
        <table>
            <thead>
                <tr>
                    <th>Check Name</th>
                    <th>Status</th>
                    <th>Details</th>
                </tr>
            </thead>
            <tbody>
                {% for r in validations %}
                <tr>
                    <td>{{ r.field_name | replace('_', ' ') | title }}</td>
                    <td style="color: {% if r.status == 'PASS' %}#28a745{% elif r.status == 'WARN' %}#ffc107{% else %}#dc3545{% endif %}; font-weight: bold;">
                        {{ r.status }}
                    </td>
                    <td>Expected: {{ r.expected }} | Actual: {{ r.actual }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        {% if fraud_signals %}
        <div class="section-title">Fraud Signals Flagged</div>
        <table>
            <thead>
                <tr>
                    <th>Detector</th>
                    <th>Signal</th>
                    <th>Risk Impact</th>
                </tr>
            </thead>
            <tbody>
                {% for s in fraud_signals %}
                <tr>
                    <td>{{ s.source }}</td>
                    <td><strong>{{ s.name }}</strong><br><small>{{ s.description }}</small></td>
                    <td style="color: #dc3545; font-weight: bold;">+{{ s.score }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
    </div>
</body>
</html>
"""
            with open(html_template_path, 'w', encoding='utf-8') as f:
                f.write(content)

    def generate_html(self, packet: DocumentPacket, out_path: str) -> str:
        """Renders the HTML template with the DocumentPacket data."""
        try:
            env = Environment(loader=FileSystemLoader(self.template_dir))
            template = env.get_template("audit_report.html")

            # Determine verdict
            verdict = "PASS"
            if packet.fraud_risk_score >= 50:
                verdict = "FAIL"
            else:
                for res in packet.validation_results:
                    if res.status == "FAIL":
                        verdict = "FAIL"
                        break
                    elif res.status == "WARN":
                        verdict = "WARN"

            from datetime import datetime
            html_content = template.render(
                verdict=verdict,
                document_type=packet.document_type.value,
                risk_score=packet.fraud_risk_score,
                image_path=os.path.basename(packet.image_path),
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                fields=packet.extracted_fields,
                validations=packet.validation_results,
                fraud_signals=packet.fraud_signals
            )

            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"HTML report generated at: {out_path}")
            return out_path
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return ""

    def generate_pdf(self, packet: DocumentPacket, out_path: str) -> str:
        """Generates a professional PDF report using ReportLab Flowables."""
        try:
            doc = SimpleDocTemplate(out_path, pagesize=letter,
                                    rightMargin=40, leftMargin=40,
                                    topMargin=40, bottomMargin=40)
            story = []
            
            # Setup styles
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle(
                'TitleStyle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor("#003366"),
                spaceAfter=15
            )
            
            section_style = ParagraphStyle(
                'SectionStyle',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor("#333333"),
                spaceBefore=15,
                spaceAfter=8,
                borderColor=colors.HexColor("#003366"),
                borderWidth=1,
                borderPadding=4
            )
            
            body_style = styles['Normal']
            
            # 1. Header Band
            story.append(Paragraph("DocuScan India Audit Report", title_style))
            story.append(Paragraph("Offline Government Identity Verification System", body_style))
            story.append(Spacer(1, 15))

            # Determine Verdict
            verdict = "PASS"
            if packet.fraud_risk_score >= 50:
                verdict = "FAIL"
            else:
                for res in packet.validation_results:
                    if res.status == "FAIL":
                        verdict = "FAIL"
                        break
                    elif res.status == "WARN":
                        verdict = "WARN"

            verdict_color = colors.HexColor("#28a745") if verdict == "PASS" else (colors.HexColor("#ffc107") if verdict == "WARN" else colors.HexColor("#dc3545"))

            # Summary Table
            summary_data = [
                ["Document Type:", packet.document_type.value, "Verdict Status:", verdict],
                ["Risk Score:", f"{packet.fraud_risk_score}/100", "Timestamp:", datetime_now_str()],
                ["Image File:", os.path.basename(packet.image_path), "", ""]
            ]
            t_summary = Table(summary_data, colWidths=[110, 150, 110, 150])
            t_summary.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8f9fa")),
                ('TEXTCOLOR', (1,0), (1,0), colors.HexColor("#003366")),
                ('TEXTCOLOR', (3,0), (3,0), verdict_color),
                ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
                ('FONTNAME', (1,0), (1,1), 'Helvetica'),
                ('FONTNAME', (3,1), (3,2), 'Helvetica'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#dee2e6")),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(t_summary)
            story.append(Spacer(1, 15))

            # 2. Extracted Fields Section
            story.append(Paragraph("Extracted Fields", section_style))
            field_data = [["Field Name", "Extracted Value", "Confidence"]]
            for k, v in packet.extracted_fields.items():
                disp_name = k.replace('_', ' ').title()
                field_data.append([disp_name, v.value, f"{int(v.confidence * 100)}%"])

            t_fields = Table(field_data, colWidths=[180, 240, 100])
            t_fields.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#003366")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#dee2e6")),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(t_fields)
            story.append(Spacer(1, 15))

            # 3. Validation Section
            story.append(Paragraph("Field Validation Rules", section_style))
            val_data = [["Rule Check", "Status", "Details"]]
            for r in packet.validation_results:
                status_p = Paragraph(f"<font color='{get_status_hex(r.status)}'><b>{r.status}</b></font>", body_style)
                val_data.append([r.field_name.replace('_', ' ').title(), status_p, f"Exp: {r.expected} | Act: {r.actual}"])

            t_val = Table(val_data, colWidths=[150, 80, 290])
            t_val.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#333333")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#dee2e6")),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8f9fa")]),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(t_val)
            story.append(Spacer(1, 15))

            # 4. Fraud Section
            if packet.fraud_signals:
                story.append(Paragraph("Fraud Risk Signal Analysis", section_style))
                fraud_data = [["Source", "Signal Description", "Risk Score"]]
                for s in packet.fraud_signals:
                    desc_para = Paragraph(f"<b>{s.name}</b><br/><font size='8' color='#666666'>{s.description}</font>", body_style)
                    fraud_data.append([s.source, desc_para, f"+{s.score}"])

                t_fraud = Table(fraud_data, colWidths=[120, 320, 80])
                t_fraud.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#c0392b")),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#dee2e6")),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f9ebea")]),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('TEXTCOLOR', (2,1), (2,-1), colors.HexColor("#c0392b")),
                    ('FONTNAME', (2,1), (2,-1), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                    ('TOPPADDING', (0,0), (-1,-1), 6),
                ]))
                story.append(t_fraud)

            # Build Document
            doc.build(story)
            logger.info(f"PDF report generated at: {out_path}")
            return out_path
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            return ""

def datetime_now_str() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_status_hex(status: str) -> str:
    if status == "PASS":
        return "#28a745"
    elif status == "WARN":
        return "#ffc107"
    return "#dc3545"
