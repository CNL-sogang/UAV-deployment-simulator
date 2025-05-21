def clear_results(self):
    self.figure_1.clear()
    self.canvas_1.draw()   
    self.figure_2_1.clear()
    self.canvas_2_1.draw()
    self.table_title.setVisible(False)  
    self.subtitle_1.setVisible(False) 
    self.subtitle_2.setVisible(False) 
    self.subtitle_3.setVisible(False) 
    for table in self.tables:
        table.clearContents()
        table.setRowCount(0)
        table.setColumnCount(0)
    for figure, canvas in zip(self.figures_3, self.canvases_3):
        figure.clear()
        canvas.draw()
