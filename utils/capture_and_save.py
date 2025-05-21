from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPainter, QPageSize, QPageLayout, QPen, QColor
from PyQt6.QtPrintSupport import QPrinter
from PyQt6.QtCore import QMarginsF, QSizeF
import os
from datetime import datetime

def capture_and_save(self):
    screen = QApplication.primaryScreen()
    if screen is None:
        print("Fail to capture.")
        return
    window_rect = self.frameGeometry()
    x, y = window_rect.topLeft().x(), window_rect.topLeft().y() 
    width, height = window_rect.width(), window_rect.height() 
    screenshot = screen.grabWindow(0, x, y, width, height)
    painter = QPainter(screenshot)
    pen = QPen(QColor(0, 0, 0))  
    pen.setWidth(1)
    painter.setPen(pen)
    painter.drawRect(0, 0, screenshot.width() - 1, screenshot.height() - 1)  
    painter.end()
    
    save_dir = self.capture_save_dir or "."
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ui_capture_{timestamp}.pdf"
    pdf_filename = os.path.abspath(os.path.join(save_dir, filename))  # 전체 경로
    
    printer = QPrinter(QPrinter.PrinterMode.HighResolution)
    printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
    printer.setOutputFileName(pdf_filename)
    printer.setResolution(72)
    page_size = QPageSize(QSizeF(width, height), QPageSize.Unit.Point)
    page_layout = QPageLayout(page_size, QPageLayout.Orientation.Portrait, QMarginsF(0, 0, 0, 0), QPageLayout.Unit.Point)
    printer.setPageLayout(page_layout)
    painter = QPainter(printer)
    painter.drawPixmap(0, 0, screenshot)  
    painter.end()

    print(f"Complete capture: {pdf_filename}")
