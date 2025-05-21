import numpy as np
import geopy.distance
import contextily as ctx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import proj3d 
from pyproj import Transformer
from scipy.interpolate import Rbf

def update_plot(self, Uopt, Vopt_k_sets):
    try:
        params = self.simulation_params
        origin_lat = params["origin_lat"]
        origin_lon = params["origin_lon"]
        distance_lat = params["distance_lat"]
        distance_lon = params["distance_lon"]
        z_min = params["z_min"]
        z_max = params["z_max"]
        P_min = params["P_min"]
        P_max = params["P_max"]
        G = params["G"]
        G_vals = params["G_vals"]
        G_vals_1 = params["G_vals_1"]
        N = params["N"]
        K = params["K"]
        f_c = params["f_c"]
        B = params["B"]
        c = params["c"]
        nsd = params["nsd"]
        environment = params["environment"]
        a = params["a"]
        b = params["b"]
        e_LoS = params["e_LoS"]
        e_NLoS = params["e_NLoS"]
        P_N = params["P_N"]
        Q = params["Q"]
        V = params["V"]
        temper = params["temper"]
        humidity = params["humidity"]
        rain = params["rain"]
        snow = params["snow"]
        pressure = params["pressure"]
        random_points = params["random_points"]
        elevations = params["elevations"]
        west = params["west"]
        east = params["east"]
        south = params["south"]
        north = params["north"]
        SINR_k_sets = self.SINR_k_sets
        best_ee_history = self.best_ee_history 
        p_history = self.p 
        
    except KeyError:
        return
    
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
    
    grid_x, grid_y = np.mgrid[lon_lat_points[:, 0].min():lon_lat_points[:, 0].max():100j, 
                            lon_lat_points[:, 1].min():lon_lat_points[:, 1].max():100j]
    
    rbf = Rbf(lon_lat_points[:, 0], lon_lat_points[:, 1], plot_elevations, function='cubic')
    grid_z = rbf(grid_x, grid_y)
    
    for i, (group, marker) in enumerate(zip(G_vals, markers)):
        indices = np.where(Q == group)[0]
        ax_2d.scatter(lon_lat_points[:,0][indices], lon_lat_points[:,1][indices], c='white', marker=marker, alpha=1, s=30)

    ax_3d = self.figure_1.add_subplot(111, projection='3d')
    
    ax_3d.get_proj = lambda: np.dot(Axes3D.get_proj(ax_3d), np.diag([1, 1, 1.3, 1]))
    
    ax_3d.set_position([0, 0, 0.8, 0.9])
    ax_3d.set_xlabel('Longitude', fontsize=10, fontweight='bold')
    ax_3d.set_ylabel('Latitude', fontsize=10, fontweight='bold')
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
    
    def xy_to_latlon(x, y, origin_lat, origin_lon):
        lat = geopy.distance.distance(meters=abs(y)).destination((origin_lat, origin_lon), 0 if y >= 0 else 180)[0]
        lon = geopy.distance.distance(meters=abs(x)).destination((origin_lat, origin_lon), 90 if x >= 0 else 270)[1]
        return lat, lon
    
    def latlon_to_xy(lat, lon, origin_lat, origin_lon):
        x = geopy.distance.distance((origin_lat, origin_lon), (origin_lat, lon)).m
        y = geopy.distance.distance((origin_lat, origin_lon), (lat, origin_lon)).m
        if lon < origin_lon:
            x = -x
        if lat < origin_lat:
            y = -y
        return x, y
        
    def PL_rain(f, d, R):
        f = f / 1e9  
        d = d / 1e3 

        co_for_k = np.array([
            [-5.33980, -0.10008, 1.13098, -0.18961, 0.71147],
            [-0.35351,  1.26970, 0.45400, 0, 0],
            [-0.23789,  0.86036, 0.15354, 0, 0],
            [-0.94158,  0.64552, 0.16817, 0, 0]
        ])

        co_for_a = np.array([
            [-0.14318, 1.82442, -0.55187, 0.67849, -1.95537],
            [0.29591, 0.77564, 0.19822, 0, 0],
            [0.32177, 0.63773, 0.13164, 0, 0],
            [-5.37610, -0.96230, 1.47828, 0, 0],
            [16.1721, -3.29980, 3.43990, 0, 0]
        ])

        logk = 0
        for j in range(4):
            logk += co_for_k[j, 0] * np.exp(-((np.log10(f) - co_for_k[j, 1]) / co_for_k[j, 2])**2)
        logk += co_for_k[0, 3] * np.log10(f) + co_for_k[0, 4]
        k = 10 ** logk

        alpha = 0
        for j in range(5):
            alpha += co_for_a[j, 0] * np.exp(-((np.log10(f) - co_for_a[j, 1]) / co_for_a[j, 2])**2)
        alpha += co_for_a[0, 3] * np.log10(f) + co_for_a[0, 4]

        gamma_r = k * R ** alpha

        L_r = gamma_r * d
        return L_r

    def PL_hum(f, d, RH, Tc, p):
        f = f / 1e9  
        d = d / 1e3  
        T = Tc + 273.15 
        theta = 300 / T
        rho = 216.7 * (((RH/100)*6.112*np.exp(17.62*Tc/(243.12+Tc))) / T) 
        e = rho * T / 216.7

        data_o = np.array([
            [50.474214, 0.975, 9.651, 6.690, 0.0, 2.566, 6.850],
            [50.987745, 2.529, 8.653, 7.170, 0.0, 2.246, 6.800],
            [51.503360, 6.193, 7.709, 7.640, 0.0, 1.947, 6.729],
            [52.021429, 14.320, 6.819, 8.110, 0.0, 1.667, 6.640],
            [52.542418, 31.240, 5.983, 8.580, 0.0, 1.388, 6.526],
            [53.066934, 64.290, 5.201, 9.060, 0.0, 1.349, 6.206],
            [53.595775, 124.600, 4.474, 9.550, 0.0, 2.227, 5.085],
            [54.130025, 227.300, 3.800, 9.960, 0.0, 3.170, 3.750],
            [54.671180, 389.700, 3.182, 10.370, 0.0, 3.558, 2.654],
            [55.221384, 627.100, 2.618, 10.890, 0.0, 2.560, 2.952]
        ])

        data_w = np.array([
            [22.235080, 0.1079, 2.144, 26.38, 0.76, 5.087, 1.00],
            [67.803960, 0.0011, 8.732, 28.58, 0.69, 4.930, 0.82],
            [119.995940, 0.0007, 8.353, 29.48, 0.70, 4.780, 0.79],
            [183.310087, 2.273, 0.668, 29.06, 0.77, 5.022, 0.85],
            [321.225630, 0.0470, 6.179, 24.04, 0.67, 4.398, 0.54]
        ])

        d_d = 5.6e-4 * (p + e) * theta**0.8
        N_D = f * p * theta**2 * ((6.14e-5) / (d_d * (1 + (f / d_d)**2)) + (1.4e-12 * p * theta**1.5) / (1 + 1.9e-5 * f**1.5))

        SF_o = 0
        for i in range(len(data_o)):
            fi = data_o[i, 0]
            delta_f_1 = data_o[i, 3] * 1e-4 * (p * theta**(0.8 - data_o[i, 4] + 1.1 * e * theta))
            delta_f_2 = np.sqrt(delta_f_1**2 + 2.25e-6)
            cofac = (data_o[i, 5] + data_o[i, 6] * theta) * 1e-4 * (p + e) * theta**0.8
            SF_o += (data_o[i, 1] * 1e-7 * p * theta**3 * np.exp(data_o[i, 2] * (1 - theta))) * (
                (f / fi) * ((delta_f_2 - cofac * (fi - f)) / ((fi - f)**2 + delta_f_2**2) +
                            (delta_f_2 - cofac * (fi + f)) / ((fi + f)**2 + delta_f_2**2)))

        SF_w = 0
        for i in range(len(data_w)):
            fi = data_w[i, 0]
            delta_f_1 = data_w[i, 3] * 1e-4 * (p * theta**(data_w[i, 4]) + data_w[i, 5] * e * theta**(data_w[i, 6]))
            delta_f_2 = 0.535 * delta_f_1 + np.sqrt(0.217 * (delta_f_1**2) + 2.1316e-12 * fi**2 / theta)
            cofac = 0
            SF_w += (data_w[i, 1] * 1e-1 * e * theta**3.5 * np.exp(data_w[i, 2] * (1 - theta))) * (
                (f / fi) * ((delta_f_2 - cofac * (fi - f)) / ((fi - f)**2 + delta_f_2**2) +
                            (delta_f_2 - cofac * (fi + f)) / ((fi + f)**2 + delta_f_2**2)))

        N_o = N_D + SF_o
        N_w = SF_w
        gamma_o = 0.1820 * f * N_o
        gamma_w = 0.1820 * f * N_w
        gamma_total = gamma_o + gamma_w

        L_h = gamma_total * d
        return L_h

    def PL_snow(f, d, R, c):
        ld = c * 100 / f 
        d = d / 1e3  
        gamma_s = 0.00349 * (R ** 1.6) / (ld ** 4) + 0.00224 * R / ld
        
        L_s = gamma_s * d
        return L_s
        
    vec_xy_to_latlon = np.vectorize(lambda x, y: xy_to_latlon(x, y, origin_lat, origin_lon))
    templat, templon = vec_xy_to_latlon(Uopt[:, 0], Uopt[:, 1])
    Uoptlonlat = np.column_stack((templon, templat, Uopt[:, 2]))
                
    Voptlonlat_sets = {}
    for k, V_k in Vopt_k_sets.items(): 
        transformed_list = []
        for v in V_k:  
            x, y, z, power = v  
            lat, lon = xy_to_latlon(x, y, origin_lat, origin_lon)  
            transformed_list.append([lon, lat, z, power])  

        Voptlonlat_sets[k] = np.array(transformed_list)
    for k in range(K):
        V_k = Voptlonlat_sets[k]
        u_k_2d = Uoptlonlat[k, :2]
        for v in V_k:
            ax_2d.plot([v[0], u_k_2d[0]], [v[1], u_k_2d[1]], 'w--', linewidth=0.8)
    
    vec_latlon_to_xy = np.vectorize(lambda lat, lon: latlon_to_xy(lat, lon, origin_lat, origin_lon))
    x_points, y_points = vec_latlon_to_xy(lon_lat_points[:, 1], lon_lat_points[:, 0])
            
    margin = 160
    x_grid, y_grid = np.meshgrid(np.linspace(-distance_lon-margin, distance_lon+margin, 100), np.linspace(-distance_lat-margin, distance_lat+margin, 100))
    rbf2 = Rbf(x_points,  y_points, plot_elevations, function='cubic')
    z_grid = rbf2(x_grid, y_grid)
    snr_grid = np.zeros_like(x_grid)

    for k in range(K):
        u_k = Uopt[k, :3]
        P_k = Uopt[k, 3]
        distances = np.sqrt((x_grid - u_k[0])**2 + (y_grid - u_k[1])**2 + (z_grid - u_k[2])**2)
        L_r_values = PL_rain(f_c, distances, rain)
        L_s_values = PL_snow(f_c, distances, snow, c)
        L_h_values = PL_hum(f_c, distances, humidity, temper, pressure)
        PL_w = L_r_values + L_s_values + L_h_values 
        fspl = 20 * np.log10(distances) + 20 * np.log10(4 * np.pi * f_c / c)
        h = np.sqrt((z_grid - u_k[2])**2)
        r = np.sqrt((x_grid - u_k[0])**2 + (y_grid - u_k[1])**2)
        elevation_angle = 180 / np.pi * np.arctan2(h, r)
        p_LoS = 1 / (1 + a * np.exp(-b * (elevation_angle - a)))
        p_NLoS = 1 - p_LoS
        PL_k = fspl + p_LoS * e_LoS + p_NLoS * e_NLoS + PL_w
        
        signal_power_lin = 10 ** ((P_k - 30 - PL_k) / 10)
        interf_power_lin = np.zeros_like(signal_power_lin)
        for j in range(K):
            if j == k:
                continue
            u_j = Uopt[j, :3] 
            P_j = Uopt[j, 3]
            distances_j = np.sqrt((x_grid - u_j[0])**2 + (y_grid - u_j[1])**2 + (z_grid - u_j[2])**2)
            L_r_values_j = PL_rain(f_c, distances_j, rain)
            L_s_values_j = PL_snow(f_c, distances_j, snow, c)
            L_h_values_j = PL_hum(f_c, distances_j, humidity, temper, pressure)
            PL_w_j = L_r_values_j + L_s_values_j + L_h_values_j 
            fspl_j = 20 * np.log10(distances_j) + 20 * np.log10(4 * np.pi * f_c / c)
            h_j = np.sqrt((z_grid - u_j[2])**2)
            r_j = np.sqrt((x_grid - u_j[0])**2 + (y_grid - u_j[1])**2)
            elevation_angle_j = 180 / np.pi * np.arctan2(h_j, r_j)
            p_LoS_j = 1 / (1 + a * np.exp(-b * (elevation_angle_j - a)))
            p_NLoS_j = 1 - p_LoS_j
            PL_j = fspl_j + p_LoS_j * e_LoS + p_NLoS_j * e_NLoS + PL_w_j 
            received_power_j = 10 ** ((P_j - 30 - PL_j) / 10)  
            interf_power_lin += received_power_j
        
        noise_power_lin = 10 ** ((P_N - 30)/ 10)  
        SINR_lin = signal_power_lin / (interf_power_lin + noise_power_lin)
        SINR_dB = 10 * np.log10(SINR_lin)
        
        snr_grid = np.maximum(snr_grid, SINR_dB)

    lat_grid, lon_grid = vec_xy_to_latlon(x_grid, y_grid)
    
    contour = ax_2d.contourf(lon_grid, lat_grid, snr_grid, levels=20, cmap='jet', alpha=0.3, antialiased=True)

    cbar_ax = inset_axes(ax_2d, width="35%", height="6%", loc="upper right")
    cbar = self.figure_2_1.colorbar(contour, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(direction="in", pad=-7, length=2, labelsize=5)  
    cbar.set_label("SINR (dB)", fontsize=6, labelpad=-16, fontweight='bold')  

    self.figure_1.subplots_adjust(left=0, right=0.95, bottom=-0.15, top=1)
    self.figure_2_1.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    ax_3d.legend(bbox_to_anchor=(0.675, 1.4), loc='upper left', borderaxespad=0, handletextpad=0)
    ax_3d.text2D(0.055, 1.23, f" ▪ Final Energy Efficiency \n      : {best_ee_history[-1]:.2e} (bits/Joule)\n ▪ Final Total Transmit Power \n      : {p_history[-1]:.2f} (dBm)", transform=ax_3d.transAxes,
            fontsize=10, bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='square,pad=0.5'))
    
    cbar_ax3d = inset_axes(ax_3d, width="35%", height="6%", loc="lower right", bbox_to_anchor=(0.05, -0.17, 1, 1), bbox_transform=ax_3d.transAxes)
    cbar = self.figure_1.colorbar(contour3dd, cax=cbar_ax3d, orientation="horizontal")
    cbar.ax.tick_params(direction="in", pad=-7, length=2, labelsize=5)  
    cbar.set_label("Elevation (m)", fontsize=6, labelpad=-15, fontweight='bold') 
    
    ax_3d.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.3f}'))
    ax_3d.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val:.3f}'))        
    
    uav_img = mpimg.imread('uav_icon.png')
    for i in range(K):
        for (x, y, z) in Uoptlonlat[:, :3]:
            x2, y2, _ = proj3d.proj_transform(x, y, z, ax_3d.get_proj())
            imagebox = OffsetImage(uav_img, zoom=0.2)
            ab = AnnotationBbox(imagebox, (x2, y2), frameon=False)
            ax_3d.add_artist(ab)
        ax_3d.plot([Uoptlonlat[i, 0], Uoptlonlat[i, 0]],[Uoptlonlat[i, 1], Uoptlonlat[i, 1]],[Uoptlonlat[i, 2], 0], color='black', linestyle=':', linewidth=1)
    uav_img_2 = mpimg.imread('uav_icon_2.png')
    for i in range(K):
        for (x, y) in Uoptlonlat[:, :2]:
            imagebox_2 = OffsetImage(uav_img_2, zoom=0.2)
            ab_2 = AnnotationBbox(imagebox_2, (x, y), frameon=False)
            ax_2d.add_artist(ab_2)
    
    self.canvas_1.draw()
    self.canvas_2_1.draw()
