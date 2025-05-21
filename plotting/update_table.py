import numpy as np
import geopy.distance
from PyQt6.QtWidgets import QTableWidgetItem

def update_table(self, Uopt):
    params = self.simulation_params
    origin_lat = params["origin_lat"]
    origin_lon = params["origin_lon"]
    def xy_to_latlon(x, y, origin_lat, origin_lon):
        lat = geopy.distance.distance(meters=abs(y)).destination((origin_lat, origin_lon), 0 if y >= 0 else 180)[0]
        lon = geopy.distance.distance(meters=abs(x)).destination((origin_lat, origin_lon), 90 if x >= 0 else 270)[1]
        return lat, lon
    vec_xy_to_latlon = np.vectorize(lambda x, y: xy_to_latlon(x, y, origin_lat, origin_lon))
    templat, templon = vec_xy_to_latlon(Uopt[:, 0], Uopt[:, 1])
    Uoptlatlon = np.column_stack((templat, templon, Uopt[:, 2], Uopt[:, 3]))
    rows, cols = Uopt.shape
    self.table_title.setText("Optimal Deployment Information")
    self.table_title.setVisible(True) 
    self.subtitle_1.setVisible(False)
    self.subtitle_2.setVisible(False)
    self.subtitle_3.setVisible(False)
    for table in self.tables:
        table.setRowCount(rows)
        table.setColumnCount(cols)
        table.setHorizontalHeaderLabels(["Latitude", "Longitude", "Height (m)", "Tx Power (dBm)"]) 
        for i in range(rows):
            for j in range(cols):
                if j < 2:
                    table.setItem(i, j, QTableWidgetItem(f"{Uoptlatlon[i, j]:.5f}")) 
                else:
                    table.setItem(i, j, QTableWidgetItem(f"{Uoptlatlon[i, j]:.1f}")) 
