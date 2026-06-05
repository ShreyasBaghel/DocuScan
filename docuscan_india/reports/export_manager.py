import os
from utils.document_packet import DocumentPacket
from reports.json_serializer import JsonSerializer
from reports.report_generator import ReportGenerator
from reports.audit_logger import AuditLogger
from utils.logger import get_logger

logger = get_logger("export_manager")

class ExportManager:
    def __init__(self):
        self.app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.export_dir = os.path.join(self.app_dir, "data", "exports")
        os.makedirs(self.export_dir, exist_ok=True)
        
        self.report_generator = ReportGenerator()
        self.audit_logger = AuditLogger()

    def export_all(self, packet: DocumentPacket) -> str:
        """
        Exports the processed packet to JSON, HTML, and PDF,
        and logs the record in the SQLite database.
        Returns the path to the generated PDF report.
        """
        # Determine base filename from uploaded document
        base_name = os.path.splitext(os.path.basename(packet.image_path))[0]
        
        json_path = os.path.join(self.export_dir, f"{base_name}_audit.json")
        html_path = os.path.join(self.export_dir, f"{base_name}_audit.html")
        pdf_path = os.path.join(self.export_dir, f"{base_name}_audit.pdf")

        # 1. Export JSON
        try:
            json_str = JsonSerializer.serialize(packet)
            with open(json_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"Saved JSON export: {json_path}")
        except Exception as e:
            logger.error(f"Failed to export JSON: {e}")

        # 2. Export HTML
        try:
            self.report_generator.generate_html(packet, html_path)
        except Exception as e:
            logger.error(f"Failed to export HTML: {e}")

        # 3. Export PDF
        pdf_report_path = ""
        try:
            pdf_report_path = self.report_generator.generate_pdf(packet, pdf_path)
            packet.report_path = pdf_report_path
        except Exception as e:
            logger.error(f"Failed to export PDF: {e}")

        # 4. Save to Audit SQLite Database
        try:
            self.audit_logger.log(packet)
        except Exception as e:
            logger.error(f"Failed to write to database audit log: {e}")

        return pdf_report_path
