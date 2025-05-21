from PyQt6.QtCore import Qt

def weather_input(self, state):
    if state == Qt.CheckState.Checked.value:
        self.w1_input_field.setEnabled(False)
        self.w1_input_field.setStyleSheet("font-family: 'Segoe UI'; background-color: lightgray") 
        self.w2_input_field.setEnabled(False)
        self.w2_input_field.setStyleSheet("font-family: 'Segoe UI'; background-color: lightgray;") 
        self.w3_input_field.setEnabled(False)
        self.w3_input_field.setStyleSheet("font-family: 'Segoe UI'; background-color: lightgray;") 
        self.w4_input_field.setEnabled(False)
        self.w4_input_field.setStyleSheet("font-family: 'Segoe UI'; background-color: lightgray;") 
    else:
        self.w1_input_field.setEnabled(True)
        self.w1_input_field.setStyleSheet("font-family: 'Segoe UI'; background-color: white; ")
        self.w2_input_field.setEnabled(True)
        self.w2_input_field.setStyleSheet("font-family: 'Segoe UI'; background-color: white;")  
        self.w3_input_field.setEnabled(True)
        self.w3_input_field.setStyleSheet("font-family: 'Segoe UI'; background-color: white;")  
        self.w4_input_field.setEnabled(True)
        self.w4_input_field.setStyleSheet("font-family: 'Segoe UI'; background-color: white; ")  
