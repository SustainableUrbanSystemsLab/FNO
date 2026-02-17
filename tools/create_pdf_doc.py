# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fpdf2",
# ]
# ///

from fpdf import FPDF
import datetime

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'FNO Model Data Specification', border=False, align='C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} - Generated on {datetime.date.today()}', align='C')

def create_pdf():
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    
    # 1. Input Tensor
    pdf.set_font("helvetica", 'B', 14)
    pdf.cell(0, 10, "1. Input Tensor (X)", ln=True)
    pdf.set_font("helvetica", size=11)
    pdf.multi_cell(0, 6, "Shape: [8, Height, Width]\nThe input consists of 8 Channels representing the physical geometry and wind conditions.")
    pdf.ln(5)
    
    # Table Header
    pdf.set_fill_color(200, 220, 255)
    pdf.set_font("helvetica", 'B', 10)
    pdf.cell(15, 8, "Idx", 1, 0, 'C', True)
    pdf.cell(30, 8, "Name", 1, 0, 'C', True)
    pdf.cell(85, 8, "Description", 1, 0, 'C', True)
    pdf.cell(35, 8, "Normalization", 1, 0, 'C', True)
    pdf.cell(25, 8, "Range (Approx)", 1, 1, 'C', True)
    
    # Table Content
    data = [
        ("0", "SDF", "Signed Distance Field (dist to nearest wall)", "/ 200.0", "0.0 - 1.0"),
        ("1", "Bldg_height", "Height of the building at this pixel", "/ 50.0", "0.0 - 1.0"),
        ("2", "Z_relative", "Height slice of the simulation (Z coord)", "/ 10.0", "0.0 - 1.0"),
        ("3", "U_over_Uref", "Background wind ratio (inlet profile)", "* 2.0", "0.2 - 2.0"),
        ("4", "X_local", "X dist from building center", "/ 500.0", "-1.0 - 1.0"),
        ("5", "Y_local", "Y dist from building center", "/ 500.0", "-1.0 - 1.0"),
        ("6", "dir_sin", "Sine of Wind Direction", "None", "-1.0 - 1.0"),
        ("7", "dir_cos", "Cosine of Wind Direction", "None", "-1.0 - 1.0"),
    ]
    
    pdf.set_font("helvetica", size=9)
    for row in data:
        pdf.cell(15, 8, row[0], 1, 0, 'C')
        pdf.cell(30, 8, row[1], 1, 0, 'L')
        pdf.cell(85, 8, row[2], 1, 0, 'L')
        pdf.cell(35, 8, row[3], 1, 0, 'C')
        pdf.cell(25, 8, row[4], 1, 1, 'C')
        
    pdf.ln(10)
    
    # 2. Target Tensor
    pdf.set_font("helvetica", 'B', 14)
    pdf.cell(0, 10, "2. Target Tensor (Y)", ln=True)
    pdf.set_font("helvetica", size=11)
    pdf.multi_cell(0, 6, "Shape: [1, Height, Width]\nThe model predicts a single scalar value representing the Wake Deficit.")
    pdf.ln(5)
    
    # Formula
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("helvetica", 'B', 12)
    pdf.cell(0, 10, "Target_Delta_U = (Mag_U - U_ref) / U_ref", 0, 1, 'C', True)
    pdf.ln(5)
    
    # Intepretation
    pdf.set_font("helvetica", 'B', 11)
    pdf.cell(0, 8, "Interpretation:", ln=True)
    pdf.set_font("helvetica", size=10)
    pdf.cell(10)
    pdf.cell(0, 6, "- 0.0: No change (Wind speed = U_ref)", ln=True)
    pdf.cell(10)
    pdf.cell(0, 6, "- -0.5: 50% drop in speed (Strong Wake)", ln=True)
    pdf.cell(10)
    pdf.cell(0, 6, "- -1.0: Stagnation (Speed = 0)", ln=True)
    pdf.ln(5)
    
    # Reconstruction
    pdf.set_font("helvetica", 'B', 11)
    pdf.cell(0, 8, "Reconstruction Formula:", ln=True)
    pdf.set_font("courier", size=10)
    pdf.cell(10)
    pdf.cell(0, 6, "Mag_U = (Prediction * U_ref) + U_ref", ln=True)
    
    output_file = "FNO_Data_Structure.pdf"
    pdf.output(output_file)
    print(f"PDF generated: {output_file}")

if __name__ == "__main__":
    create_pdf()
