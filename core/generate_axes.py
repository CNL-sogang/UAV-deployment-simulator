import numpy as np
import geopy.distance
import contextily as ctx
import requests
import random
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QProgressDialog, QMessageBox
from PyQt6.QtCore import Qt, QTimer
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyproj import Transformer
from scipy.interpolate import Rbf

def generate_axes(self):
    progress_dialog = QProgressDialog("Terrain Loading...", None, 0, 0, self)
    progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    progress_dialog.setWindowTitle("Terrain Loading...")
    progress_dialog.setMinimumDuration(500)
    progress_dialog.setValue(0)
    progress_dialog.show()
    QTimer.singleShot(3000, progress_dialog.close)  
    try:
        origin_lat = float(self.inputs[0].text())
        origin_lon = float(self.inputs[1].text())
        distance_lat = float(self.inputs[2].text()) / 2
        distance_lon = float(self.inputs[3].text()) / 2
        z_min = float(self.inputs[4].text())
        z_max = float(self.inputs[5].text())
        P_min = float(self.inputs[6].text())
        P_max = float(self.inputs[7].text())
        G = int(self.option_select_6.currentText())    
        q_values = self.inputs[8].text().split(',')
        G_vals_1 = []
        for g in range(G):
            value = float(q_values[g].strip())
            G_vals_1.append(value)
        G_vals_1 = np.array(G_vals_1)
        N = int(self.inputs[9].text())
        K = int(self.inputs[10].text())
        f_c = float(self.inputs[11].text())
        B = float(self.inputs[12].text())
        c = float(self.inputs[13].text())
        nsd = float(self.inputs[14].text())
        environment = self.option_select.currentText()
        if environment == "Suburban":
            a, b, e_LoS, e_NLoS = 4.88, 0.43, 0.1, 21
        elif environment == "Urban":
            a, b, e_LoS, e_NLoS = 9.61, 0.16, 1, 20
        elif environment == "Dense Urban":
            a, b, e_LoS, e_NLoS = 12.08, 0.11, 1.6, 23
        elif environment == "Highrise Urban":
            a, b, e_LoS, e_NLoS = 27.23, 0.08, 2.3, 34
        if self.new_input_field.text() == "None" or not self.random_seed_checkbox.isChecked():
            seed = None
        else:
            seed = int(self.new_input_field.text())
        
    except ValueError:
        return
    
    np.random.seed(seed)
    
    P_N = nsd + 10 * np.log10(B) 
    
    reqSINR_lin = 2**(G_vals_1 / B) - 1 
    reqSINR_dB = 10 * np.log10(reqSINR_lin)
    G_vals = reqSINR_dB

    group_sizes = [N // G] * G
    for i in range(N % G):
        group_sizes[i] += 1

    Q = []
    for value, size in zip(G_vals, group_sizes):
        Q.extend([value] * size)

    np.random.shuffle(Q)
    Q = np.array(Q).reshape(-1, 1)
    
    north = geopy.distance.distance(meters=distance_lat).destination((origin_lat, origin_lon), 0)[0]
    south = geopy.distance.distance(meters=distance_lat).destination((origin_lat, origin_lon), 180)[0]
    east = geopy.distance.distance(meters=distance_lon).destination((origin_lat, origin_lon), 90)[1]
    west = geopy.distance.distance(meters=distance_lon).destination((origin_lat, origin_lon), 270)[1]
    
    random_points = []
    for _ in range(N):
        rand_lat = random.uniform(south, north)
        rand_lon = random.uniform(west, east)
        random_points.append((rand_lat, rand_lon))
        
    url = "https://www.elevation-api.eu/v1/elevation"
    locations = str([[lat, lon] for lat, lon in random_points])
    response = requests.get(url, params={"pts": locations})
    if response.status_code == 200:
        data = response.json() 
        if "elevations" in data and isinstance(data["elevations"], list) and len(data["elevations"]) == len(random_points):
            elevations = data["elevations"] 
        else:
            print("Warning: Unexpected API response format")
            elevations = [0] * len(random_points) 
    else:
        print(f"Error fetching elevation data: {response.status_code}")
        elevations = [0] * len(random_points)
    APIkey = self.api_key
    if self.weather_checkbox.isChecked():
        if not APIkey:
            QMessageBox.critical(self, "Error", "OpenWeatherMap API Key does not exist")
            return
        try:
            lat = self.inputs[0].text()
            lon = self.inputs[1].text()
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={APIkey}"
            response = requests.get(url)
            if response.status_code != 200:
                QMessageBox.warning(self, "Error", "Failed to get weather information.")
                return
            data = response.json()
            temper = data['main']['temp'] - 273.15
            humidity = data['main']['humidity']
            rain = data.get('rain', {}).get('1h', 0)  
            snow = data.get('snow', {}).get('1h', 0) 
            weather = data["weather"][0]["description"]  
            wind_speed = data["wind"]["speed"]  
            wind_deg = data["wind"]["deg"]  
            pressure = data["main"]["pressure"]
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing weather data:\n{e}")
            return
    else:
        try:
            temper = float(self.w1_input_field.text())
            humidity = float(self.w2_input_field.text())
            rain = float(self.w3_input_field.text())
            snow = float(self.w4_input_field.text())
            weather = "Unknown"
            wind_speed = "Unknown"
            wind_deg = "Unknown"
            pressure = 1013.25
        except ValueError:
            QMessageBox.warning(self, "Error", "Weather parameters are blank or not in numeric format.")
            return
        
    Winfo = np.array([[temper, humidity, rain, snow]])
    W2 = np.array([[weather, wind_speed, wind_deg, pressure]])
    
    def latlon_to_xy(lat, lon, origin_lat, origin_lon):
        x = geopy.distance.distance((origin_lat, origin_lon), (origin_lat, lon)).m
        y = geopy.distance.distance((origin_lat, origin_lon), (lat, origin_lon)).m
        if lon < origin_lon:
            x = -x
        if lat < origin_lat:
            y = -y
        return x, y

    xy_points = np.array([latlon_to_xy(lat, lon, origin_lat, origin_lon) for lat, lon in random_points])
    V = np.column_stack((xy_points[:,0], xy_points[:,1], elevations, Q))
    
    self.figure_1.clear()
    self.figure_2_1.clear()
    
    colors = ['b', 'y', 'r', 'c', 'g', 'k']
    markers = ['o', '^', 's', '*', 'D', 'X']
    
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    transformer_back = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    plot2_xy_points = np.array([transformer.transform(lon, lat) for lat, lon in random_points])
    lon_lat_points = np.array([transformer_back.transform(x, y) for x, y in plot2_xy_points])
    plot_elevations = np.array(elevations)

    ax_2d = self.figure_2_1.add_subplot(111)
    ax_2d.set_xlim(west, east)
    ax_2d.set_ylim(south, north)
    ctx.add_basemap(ax_2d, crs="EPSG:4326", source=ctx.providers.Esri.WorldImagery)

    grid_x, grid_y = np.mgrid[lon_lat_points[:, 0].min():lon_lat_points[:, 0].max():200j, 
                            lon_lat_points[:, 1].min():lon_lat_points[:, 1].max():200j]

    rbf = Rbf(lon_lat_points[:, 0], lon_lat_points[:, 1], plot_elevations, function='cubic')
    grid_z = rbf(grid_x, grid_y)
    ax_2d.contour(grid_x, grid_y, grid_z, levels=20, cmap='terrain', alpha=0.7)
    
    for i, (group, color, marker) in enumerate(zip(G_vals, colors, markers)):
        indices = np.where(Q == group)[0]
        ax_2d.scatter(lon_lat_points[:,0][indices], lon_lat_points[:,1][indices], c=color, marker=marker, alpha=1, s=30)

    ax_3d = self.figure_1.add_subplot(111, projection='3d')
    
    ax_3d.get_proj = lambda: np.dot(Axes3D.get_proj(ax_3d), np.diag([1, 1, 1.3, 1]))
    
    ax_3d.set_position([0, 0, 0.8, 0.9])
    ax_3d.set_xlabel('Longitude', fontsize=9, fontweight='bold')
    ax_3d.set_ylabel('Latitude', fontsize=9, fontweight='bold')
    ax_3d.tick_params(axis='both', which='major', labelsize=8) 
    ax_3d.set_xlim(west, east)        
    ax_3d.set_ylim(south, north)
    ax_3d.set_zlim(0, z_max+100) 
    ax_3d.set_box_aspect([1, 1, 1])  
    ax_3d.grid(False)  
    
    x_edges = [east, west, west]
    y_edges = [north, north, south]

    z_edges = list(range(100, int(z_max) + 1, 100))
    for z in z_edges:
        ax_3d.plot(x_edges, y_edges, [z] * 3, color='lightgray', linestyle=':', lw=1)
    
    contour3dd = ax_3d.plot_surface(grid_x, grid_y, grid_z, cmap='terrain', alpha=0.3)
    for i, (group, color, marker) in enumerate(zip(G_vals, colors, markers)):
        indices = np.where(Q == group)[0]
        ax_3d.scatter(lon_lat_points[:,0][indices], lon_lat_points[:,1][indices], plot_elevations[indices]+10, 
                        c=color, label=f'QoS:{G_vals_1[i]/1e6}Mbps', marker=marker, s=30, depthshade=False)
    
    self.figure_1.subplots_adjust(left=0, right=0.95, bottom=-0.15, top=1)
    self.figure_2_1.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    ax_3d.legend(bbox_to_anchor=(0.675, 1.4), loc='upper left', borderaxespad=0, handletextpad=0)
    
    cbar_ax3d = inset_axes(ax_3d, width="35%", height="6%", loc="lower right", bbox_to_anchor=(0.05, -0.17, 1, 1), bbox_transform=ax_3d.transAxes)
    cbar = self.figure_1.colorbar(contour3dd, cax=cbar_ax3d, orientation="horizontal")
    cbar.ax.tick_params(direction="in", pad=-7, length=2, labelsize=5)  
    cbar.set_label("Elevation (m)", fontsize=6, labelpad=-15, fontweight='bold') 
    
    ax_3d.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.3f}'))
    ax_3d.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.3f}'))
    
    self.canvas_1.draw()
    self.canvas_2_1.draw()
    
    self.run_button.setEnabled(True)
    self.reset_button.setEnabled(True)
    self.simulation_params = {
        "origin_lat": origin_lat, "origin_lon": origin_lon, "distance_lat": distance_lat, "distance_lon": distance_lon,
        "z_min": z_min, "z_max": z_max, "P_min": P_min, "P_max": P_max,
        "G": G, "G_vals" : G_vals, "G_vals_1" : G_vals_1,"N": N, "K": K,
        "f_c": f_c, "B": B, "c": c, "nsd": nsd, "environment": environment,
        "a": a, "b": b, "e_LoS": e_LoS, "e_NLoS": e_NLoS, "seed": seed,
        "P_N" : P_N, "x_coords" : xy_points[:,0], "y_coords" : xy_points[:,1], "z_coords" : elevations,
        "Q" : Q, "V" : V, "random_points" : random_points, "elevations" : elevations,
        "west" : west, "east" : east, "south" : south, "north" : north,
        "temper" : temper, "humidity" : humidity, "rain" : rain, "snow": snow, "weather": weather, "wind_speed": wind_speed, "wind_deg": wind_deg, "pressure": pressure
    }
    
    for i, (figure, canvas) in enumerate(zip(self.figures_3, self.canvases_3)):
        figure.clear()
        if i == 0:
            ax = figure.add_subplot(111)
            ax.set_title("Energy Efficiency (bits/Joule)\n", fontweight='bold', fontname='Segoe UI', fontsize=11)
            ax.set_xlabel("Iteration Progress", fontname='Segoe UI', fontsize=9)
            ax.minorticks_on()
            ax.tick_params(which='both', direction='in', top=True, right=True, left=True, bottom=True)
            ax.set_xticklabels([]) 
            ax.set_yticklabels([])
            figure.tight_layout()
        else:
            ax1 = figure.add_subplot(111)
            ax1.set_title("Total Transmit Power (dBm)\n", fontweight='bold', fontname='Segoe UI', fontsize=11)
            ax1.set_xlabel("Iteration Progress", fontname='Segoe UI', fontsize=9)
            ax1.minorticks_on()
            ax1.tick_params(which='both', direction='in', top=True, right=True, left=True, bottom=True)
            ax1.set_xticklabels([]) 
            ax1.set_yticklabels([]) 
            figure.tight_layout()
        canvas.draw()

    self.update_weather(Winfo, W2)
