#generate a pdf report
from fpdf import FPDF
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

class GenerateReport:
# initalize an empty pdf object
    def __init__(self, path):
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.add_page()
        self.path = path
        self.pdf.set_font("Arial", size=12)
        self.pdf.set_fill_color(200, 220, 255)
        self.pdf.set_text_color(0)
        self.pdf.set_draw_color(0)
        self.pdf.set_line_width(0.3)
        self.pdf.set_xy(10, 10)
        self.pdf.cell(0, 10, "Report", 0, 1, 'C', 1)
        self.pdf.ln(10)
        print("Creating PDF")
# add a title to the pdf
    def add_title(self, title):
        print(f"Adding title: {title}")
        self.pdf.set_font("Arial", size=16)
        self.pdf.cell(0, 10, title, 0, 1, 'L')
        self.pdf.ln(10)
# add a paragraph to the pdf
    def add_paragraph(self, paragraph):
        self.pdf.set_font("Arial", size=12)
        self.pdf.multi_cell(0, 10, paragraph)
        self.pdf.ln(10)
# add a table to the pdf
    def add_table(self, data):
        self.pdf.set_font("Arial", size=12)
        for row in data:
            for cell in row:
                self.pdf.cell(40, 10, str(cell), 1)
            self.pdf.ln()
        self.pdf.ln(10)
# add a plot to the pdf
    def add_plot(self, plot):
        self.pdf.image(plot, x = 10, y = None, w = 190, h = 100)
        self.pdf.ln(10)
# save the pdf

    def save(self):
        self.pdf.output(self.path)
        print(f"Report saved to {self.path}")