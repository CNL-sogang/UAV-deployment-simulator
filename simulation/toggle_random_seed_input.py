from PyQt6.QtCore import Qt

def toggle_random_seed_input(self, state):
    if state == Qt.CheckState.Checked.value:
        self.new_input_field.setEnabled(True)
        self.new_input_field.setStyleSheet("font-family: 'Segoe UI'; background-color: white;") 
    else:
        self.new_input_field.setEnabled(False)
        self.new_input_field.setStyleSheet("font-family: 'Segoe UI'; background-color: lightgray; border: 1px lightgray;")  
    