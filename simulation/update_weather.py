from PyQt6.QtWidgets import QTableWidgetItem
from PyQt6.QtGui import QFont

def update_weather(self, Winfo, W2):
    params = self.simulation_params
    origin_lat = params["origin_lat"]
    origin_lon = params["origin_lon"]
    directions = [
        "North", "North-Northeast", "Northeast", "East-Northeast",
        "East", "East-Southeast", "Southeast", "South-Southeast",
        "South", "South-Southwest", "Southwest", "West-Southwest",
        "West", "West-Northwest", "Northwest", "North-Northwest"
    ]
    if W2[0,2] == "Unknown":
        direction = ""
    else:
        direction = directions[round(float(W2[0,2]) / 22.5) % 16]
    rows, cols = Winfo.shape
    self.table_title.setText("Real-Time Geolocation & Meteorological Data")
    self.table_title.setVisible(True) 
    self.subtitle_1.setText(f"Latitude : <b>{origin_lat}</b>,  Longitude : <b>{origin_lon}</b>")
    self.subtitle_2.setText(f"Status : <b>{W2[0,0]}</b>,  Pressure : <b>{W2[0,3]}</b>hPa")
    self.subtitle_3.setText(f"Wind : <b>{W2[0,1]}</b>m/s; <b>{direction}</b>")
    self.subtitle_1.setVisible(True)
    self.subtitle_2.setVisible(True)
    self.subtitle_3.setVisible(True)
    for table in self.tables:
        table.setRowCount(rows)
        table.setColumnCount(cols)
        table.setHorizontalHeaderLabels(["Temperature (°C)", "Humidity (%)", "Rainrate (mm/h)", "Snowrate (mm/h)"]) 
        bold_font = QFont()
        bold_font.setBold(True)
        for i in range(rows):
            for j in range(cols):
                item = QTableWidgetItem(f"{Winfo[i, j]:.2f}") 
                item.setFont(bold_font)
                self.table_2_2.setItem(i, j, item)
    