def toggle_region3(self):
    if not self.sub_frame_3.isVisible():  
        self.sub_frame_3.setVisible(True)
        self.resize(1780, 671)
        self.result_layout.setColumnStretch(2, 17)  
    else:
        self.sub_frame_3.setVisible(False)
        self.resize(1370, 671)
        self.result_layout.setColumnStretch(2, 0)  
    