from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

def create_pdf_report(
    filename,
    patient_data,
    risk_score,
    risk_level,
    ai_report,
    suggestions
):

    doc = SimpleDocTemplate(
        filename,
        pagesize=letter
    )

    styles = getSampleStyleSheet()

    elements = []

    title = Paragraph(
        "AI Healthcare Risk Report",
        styles['Title']
    )

    elements.append(title)
    elements.append(Spacer(1, 12))

    patient_info = f"""
    <b>Age:</b> {patient_data['Age']}<br/>
    <b>Blood Pressure:</b> {patient_data['BloodPressure']}<br/>
    <b>Glucose:</b> {patient_data['Glucose']}<br/>
    """

    elements.append(
        Paragraph(patient_info, styles['BodyText'])
    )

    elements.append(Spacer(1, 12))

    risk_text = f"""
    <b>Risk Score:</b> {risk_score:.2f}%<br/>
    <b>Risk Level:</b> {risk_level}
    """

    elements.append(
        Paragraph(risk_text, styles['BodyText'])
    )

    elements.append(Spacer(1, 12))

    elements.append(
        Paragraph(
            f"<b>AI Medical Report:</b><br/>{ai_report}",
            styles['BodyText']
        )
    )

    elements.append(Spacer(1, 12))

    suggestion_text = "<br/>".join(
        [f"• {s}" for s in suggestions]
    )

    elements.append(
        Paragraph(
            f"<b>Suggestions:</b><br/>{suggestion_text}",
            styles['BodyText']
        )
    )

    elements.append(Spacer(1, 20))

    disclaimer = """
    <font color='red'>
    This report is AI-generated and for educational purposes only.
    Consult healthcare professionals for medical advice.
    </font>
    """

    elements.append(
        Paragraph(disclaimer, styles['BodyText'])
    )

    doc.build(elements)