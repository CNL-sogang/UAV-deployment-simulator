def reset_to_initial(self):
    default_values = ["37.5509442", "126.9410023", "3000", "3000", "100", "500", 
                        "20", "40", "1e6,2e6,4e6", "50", "3", "2.1e9", "10e6", "3e8", "-174"]
    for input_field, default in zip(self.inputs, default_values):
        input_field.setText(default)
    self.option_select.setCurrentText("Urban")  
    self.option_select_6.setCurrentText("3")  
    self.run_button.setEnabled(False)  
    self.reset_button.setEnabled(False)  
    self.generate_button.setEnabled(True)
    self.clear_results()
