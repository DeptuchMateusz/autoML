import unittest
import pandas as pd
from project.reporting.generate_report import GenerateReport

class TestGenerateReport(unittest.TestCase):

    # This method will run before each test
    def setUp(self):
        # Initialize the GenerateReport object
        self.report = GenerateReport("test_report.pdf")
    
    def test_add_title(self):
        # Testing if the function correctly adds a title
        self.report.add_title("Test Title")
        self.assertTrue("Test Title")
    
    def test_add_paragraph(self):
        # Testing if the function correctly adds a paragraph
        self.report.add_paragraph("Test Paragraph")
        self.assertTrue("Test Paragraph")
    
    def test_add_table(self):
        # Testing if the function correctly adds a table
        data = [[1, 2, 3], [4, 5, 6]]
        self.report.add_table(data)
        self.assertTrue(data)

    def test_save(self):
        # Testing if the function correctly saves the pdf
        self.report.save()
        self.assertTrue("test_report.pdf")

    def test_add_plot(self):
        # Testing if the function correctly adds a plot
        self.report.add_plot("tests/tests_reporting/test_plot.png")
        self.assertTrue("tests/tests_reporting/test_plot.png")

    def test_all(self):
        # Testing if the function correctly adds a title, paragraph, table, plot and saves the pdf
        self.report.add_title("Test Title")
        self.report.add_paragraph("Test Paragraph")
        data = [[1, 2, 3], [4, 5, 6]]
        self.report.add_table(data)
        self.report.add_plot("tests/tests_reporting/test_plot.png")
        self.report.save()
        self.assertTrue("Test Title")
        self.assertTrue("Test Paragraph")
        self.assertTrue(data)
        self.assertTrue("tests/tests_reporting/test_plot.png")
        self.assertTrue("test_report.pdf")

if __name__ == '__main__':
    unittest.main()