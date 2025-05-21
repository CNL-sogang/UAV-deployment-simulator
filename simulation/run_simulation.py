import numpy as np
from PyQt6.QtWidgets import QApplication, QProgressDialog
from PyQt6.QtCore import Qt, QTimer, QThread
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_simulation(self):
    try:            
        params = self.simulation_params
        distance_lat = params["distance_lat"]
        distance_lon = params["distance_lon"]
        z_min = params["z_min"]
        z_max = params["z_max"]
        P_min = params["P_min"]
        P_max = params["P_max"]
        K = params["K"]
        f_c = params["f_c"]
        B = params["B"]
        c = params["c"]
        a = params["a"]
        b = params["b"]
        e_LoS = params["e_LoS"]
        e_NLoS = params["e_NLoS"]
        seed = params["seed"]
        P_N = params["P_N"]
        Q = params["Q"]
        V = params["V"]
        temper = params["temper"]
        humidity = params["humidity"]
        rain = params["rain"]
        snow = params["snow"]
        pressure = params["pressure"]
        self.hyperparams["SA_cooling_rate"] = float(self.inputs3[0].text())
        self.hyperparams["SA_threshold_factor"] = float(self.inputs3[1].text())
        self.hyperparams["SA_iterations"] = int(self.inputs3[2].text())
        self.hyperparams["SA_no_update_count"] = int(self.inputs3[3].text())
        self.hyperparams["SA_adaptive_factor"] = float(self.inputs3[4].text())
        w_values = self.inputs3[5].text().split(',')
        self.hyperparams["w_initial"] = float(w_values[0].strip())
        self.hyperparams["w_final"] = float(w_values[1].strip())
        c_values = self.inputs3[6].text().split(',')
        self.hyperparams["PSO_c1"] = float(c_values[0].strip())
        self.hyperparams["PSO_c2"] = float(c_values[1].strip())
        self.hyperparams["PSO_particles"] = int(self.inputs3[7].text())
        self.hyperparams["PSO_iterations"] = int(self.inputs3[8].text())
        hyperparams = self.hyperparams
        cooling_rate = hyperparams["SA_cooling_rate"]
        w_initial = hyperparams["w_initial"]
        w_final = hyperparams["w_final"]
        min_threshold = hyperparams["SA_threshold_factor"]
        c1 = hyperparams["PSO_c1"]
        c2 = hyperparams["PSO_c2"]
        V_max = 0.1 * (P_max - P_min)
        max_sa_iter = hyperparams["SA_iterations"]
        max_pso_iter = hyperparams["PSO_iterations"]
        no_update_threshold = hyperparams["SA_no_update_count"]
        adaptive_factor = hyperparams["SA_adaptive_factor"]
        num_particles = hyperparams["PSO_particles"]
    except KeyError:
        return
    
    progress_dialog = QProgressDialog("Optimization in progress...", "cancel", 0, 100, self)
    progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    progress_dialog.setWindowTitle("Running Optimization")
    progress_dialog.setMinimumDuration(500)  
    progress_dialog.setValue(0)
    progress_dialog.setAutoClose(False)
    progress_dialog.show()
    
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
    
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(np.column_stack((V[:,0],V[:,1], Q)))

    kmeans_weighted = KMeans(n_clusters=K, random_state=seed, n_init=10)
    kmeans_weighted.fit(data_normalized)
    labels_weighted = kmeans_weighted.labels_
    cluster_centers_xy = scaler.inverse_transform(kmeans_weighted.cluster_centers_)[:, :2]  
    cluster_centers_z_dBscale = scaler.inverse_transform(kmeans_weighted.cluster_centers_)[:, 2]  
    cluster_centers_z = z_max - (cluster_centers_z_dBscale - min(Q)) / (max(Q) - min(Q)) * (z_max - z_min) 
    cluster_centers_z = cluster_centers_z.reshape(-1, 1)
    u_vector = np.hstack((cluster_centers_xy, cluster_centers_z)) 
    P = np.ones(K) * (P_max)
    P = P.reshape(-1, 1)  
    U = np.hstack((u_vector, P))
    Uopt = np.copy(U)
    
    def update_ue_uav_mapping():
        global V_k_sets  
        D = np.linalg.norm(V[:, np.newaxis, :3] - U[:, :3], axis=2) 
        nearest_UAV_indices = np.argmin(D, axis=1) 
        V_k_sets = {k: V[nearest_UAV_indices == k] for k in range(K)} 
        
    update_ue_uav_mapping() 

    def objective_function(P_vars):
        global V_k_sets, C_k_sets
        U[:, 3] = P_vars  
        W_k_sets = []
        C_k_sets = []

        for k in range(K):
            V_k = V_k_sets[k] 
            u_k = U[k, :3]  
            P_k = U[k, 3]  

            distances = np.linalg.norm(V_k[:, :3] - u_k, axis=1) 
            L_r_values = PL_rain(f_c, distances, rain)
            L_s_values = PL_snow(f_c, distances, snow, c)
            L_h_values = PL_hum(f_c, distances, humidity, temper, pressure)
            PL_w = L_r_values + L_s_values + L_h_values 
            fspl = 20 * np.log10(distances) + 20 * np.log10(4 * np.pi * f_c / c)
            h = np.linalg.norm(u_k[2] - V_k[:, 2])
            r = np.linalg.norm(V_k[:, :2] - u_k[:2], axis=1)  
            elevation_angle = 180 / np.pi * np.arctan2(h, r)   
            p_LoS = 1 / (1 + a * np.exp(-b * (elevation_angle - a))) 
            p_NLoS = 1 - p_LoS  
            PL_k = fspl + p_LoS * e_LoS + p_NLoS * e_NLoS + PL_w 

            signal_power_lin = 10 ** ((P_k - 30 - PL_k) / 10)
            interf_power_lin = np.zeros_like(signal_power_lin)
            for j in range(K):
                if j == k:
                    continue
                u_j = U[j, :3] 
                P_j = U[j, 3]
                distances_j = np.linalg.norm(V_k[:, :3] - u_j, axis=1)
                L_r_values_j = PL_rain(f_c, distances_j, rain)
                L_s_values_j = PL_snow(f_c, distances_j, snow, c)
                L_h_values_j = PL_hum(f_c, distances_j, humidity, temper, pressure)
                PL_w_j = L_r_values_j + L_s_values_j + L_h_values_j 
                fspl_j = 20 * np.log10(distances_j) + 20 * np.log10(4 * np.pi * f_c / c)
                h_j = np.linalg.norm(u_j[2] - V_k[:, 2])
                r_j = np.linalg.norm(V_k[:, :2] - u_j[:2], axis=1) 
                elevation_angle_j = 180 / np.pi * np.arctan2(h_j, r_j)
                p_LoS_j = 1 / (1 + a * np.exp(-b * (elevation_angle_j - a)))
                p_NLoS_j = 1 - p_LoS_j
                PL_j = fspl_j + p_LoS_j * e_LoS + p_NLoS_j * e_NLoS + PL_w_j
                received_power_j = 10 ** ((P_j - 30 - PL_j) / 10)  
                interf_power_lin += received_power_j
                noise_power_lin = 10 ** ((P_N - 30) / 10)  
                SINR_lin = signal_power_lin / (interf_power_lin + noise_power_lin)
                SINR_dB = 10 * np.log10(SINR_lin)
                SINR_k = SINR_dB
                W_k_sets.append(SINR_dB)

            mask = V_k[:, 3] <= SINR_k
            C_k = V_k[mask]
            C_k_sets.append(C_k)
            
        valid_coverage = all(len(C_k) == len(V_k_sets[k]) for k, C_k in enumerate(C_k_sets)) 
        if not valid_coverage:
            return -np.inf, 0 

        final_result = sum(
        sum(B * np.log2(1 + 10 ** (W_k_sets[k][i] / 10)) for i in range(len(C_k))) / len(C_k) 
        / (10 ** ((U[k, 3] - 30) / 10)) 
        for k, C_k in enumerate(C_k_sets)
        if len(C_k) > 0 and len(W_k_sets[k]) >= len(C_k)
        )
        return final_result / K, final_result / K  
    
    def run_pso():
        np.random.seed(None)
        particles = np.random.uniform(P_min, P_max, (num_particles, K))
        velocities = np.zeros_like(particles)
        personal_best = np.copy(particles)
        personal_best_scores = np.full(num_particles, -np.inf)
        global_best = np.copy(particles[0])
        global_best_score = -np.inf

        for iter in range(max_pso_iter):
            w = w_initial - (w_initial - w_final) * (iter / max_pso_iter) 
            for i in range(num_particles):
                score, _ = objective_function(particles[i])
                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best[i] = np.copy(particles[i])
                if score > global_best_score:
                    global_best_score = score
                    global_best = np.copy(particles[i])

            r1, r2 = np.random.rand(num_particles, K), np.random.rand(num_particles, K)
    
            for i in range(num_particles):
                velocities[i] = (
                    w * velocities[i] 
                    + c1 * r1[i] * (personal_best[i] - particles[i]) 
                    + c2 * r2[i] * (global_best - particles[i])
                )
                velocities[i] = np.clip(velocities[i], -V_max, V_max) 
                particles[i] += velocities[i] 
                particles[i] = np.clip(particles[i], P_min, P_max)
        
        return global_best, global_best_score
    
    initial_positions = np.copy(U[:, :3])
    fixed_ini_positions = np.copy(initial_positions)
    initial_P_T = np.copy(U[:, 3])
    initial_EE, _ = objective_function(initial_P_T)
    step_size_xy = (distance_lon + distance_lat)/10
    step_size_z = z_max/10
    while initial_EE == -np.inf or initial_EE <= 0:
        np.random.seed(None)
        new_positions = np.copy(fixed_ini_positions)
        new_positions[:, :2] += np.random.uniform(-step_size_xy, step_size_xy, size=(K, 2)) 
        new_positions[:, 2] += np.random.uniform(-step_size_z, step_size_z, size=K)

        new_positions[:, 0] = np.clip(new_positions[:, 0], -distance_lon, distance_lon)
        new_positions[:, 1] = np.clip(new_positions[:, 1], -distance_lat, distance_lat)
        new_positions[:, 2] = np.clip(new_positions[:, 2], z_min, z_max)
        
        P = np.random.uniform(P_min, P_max, K)
        U[:, :3] = new_positions
        U[:, 3] = P
        Uopt = np.copy(U)
        update_ue_uav_mapping()
        initial_positions = np.copy(U[:, :3])
        initial_P_T = np.copy(U[:, 3])
        initial_EE, _ = objective_function(initial_P_T)
        
    best_positions = initial_positions
    best_P_T = initial_P_T
    best_EE = initial_EE
    global_ee_max = initial_EE
    temperature = initial_EE  
    no_update_count = 0  
    total_iter_count = np.ceil(np.log10(min_threshold) / np.log10(cooling_rate)) * max_sa_iter
    ee_history = []
    best_ee_history = []
    p_history = []
    iter_count = 0 
    trial_count = 0
    while temperature >= (min_threshold * initial_EE):

        np.random.seed(None) 

        for _ in range(max_sa_iter):
            new_positions = np.copy(best_positions)
            new_positions[:, :2] = best_positions[:, :2] + np.random.uniform(-step_size_xy, step_size_xy, size=(K, 2)) 
            new_positions[:, 2] = best_positions[:, 2] + np.random.uniform(-step_size_z, step_size_z, size=K) 
            new_positions[:, 0] = np.clip(new_positions[:, 0], -distance_lon, distance_lon)
            new_positions[:, 1] = np.clip(new_positions[:, 1], -distance_lat, distance_lat)
            new_positions[:, 2] = np.clip(new_positions[:, 2], z_min, z_max)

            U[:, :3] = new_positions  
            update_ue_uav_mapping()  
            new_P_T, new_EE = run_pso()
            
            while new_EE == -np.inf or initial_EE <= 0 :
                np.random.seed(None)
        
                new_positions[:, :2] = best_positions[:, :2] + np.random.uniform(-step_size_xy, step_size_xy, size=(K, 2)) 
                new_positions[:, 2] = best_positions[:, 2] + np.random.uniform(-step_size_z, step_size_z, size=K) 

                new_positions[:, 0] = np.clip(new_positions[:, 0], -distance_lon, distance_lon)
                new_positions[:, 1] = np.clip(new_positions[:, 1], -distance_lat, distance_lat)
                new_positions[:, 2] = np.clip(new_positions[:, 2], z_min, z_max)

                U[:, :3] = new_positions
                update_ue_uav_mapping()  
                new_P_T, new_EE = run_pso()
    
            delta_EE = new_EE - best_EE  

            if delta_EE > 0 :
                best_positions = np.copy(new_positions)
                best_EE = new_EE
                best_P_T = new_P_T
                if new_EE > global_ee_max :
                    Uopt[:, :3] = best_positions
                    Uopt[:, 3] = best_P_T
                    trial_count = 1
                no_update_count = 0
            
            elif np.exp(delta_EE / temperature) > np.random.rand() :
                best_positions = np.copy(new_positions)
                best_EE = new_EE
                best_P_T = new_P_T 
                no_update_count = 0
    
            elif no_update_count >= no_update_threshold :
                if trial_count == 1 :
                    best_positions = np.copy(new_positions)
                    best_EE = new_EE
                    best_P_T = new_P_T
                    no_update_count = 0
                    trial_count = 0
                else:
                    step_size_xy = min(step_size_xy * adaptive_factor , (distance_lon+distance_lat)/2/2)
                    step_size_z = min(step_size_z * adaptive_factor , z_max/2)
                    no_update_count = 0
            else:
                no_update_count += 1  
        
            temp = 10* np.log10(np.sum(10 ** ((best_P_T / 10)))) 
            ee_history.append(best_EE)
            p_history.append(temp)
            if best_ee_history:
                global_ee_max = max(best_ee_history)
                best_ee_history.append(max(best_ee_history[-1], best_EE))
            else:
                best_ee_history.append(best_EE)
            
            iter_count += 1  
            progress = int((iter_count + 1) / total_iter_count * 100)
            progress_dialog.setValue(progress)

            QApplication.processEvents()
    
            if progress_dialog.wasCanceled():
                return  

        temperature *= cooling_rate  
    
    for i in range(progress, 101):
        progress_dialog.setValue(i)
        QApplication.processEvents()
        QThread.msleep(5) 
        
    self.ee_history = ee_history
    self.best_ee_history = best_ee_history
    self.p = p_history
    
    def update_final_mapping():
        global Vopt_k_sets  
        D = np.linalg.norm(V[:, np.newaxis, :3] - Uopt[:, :3], axis=2)  
        nearest_UAV_indices = np.argmin(D, axis=1)  
        Vopt_k_sets = {k: V[nearest_UAV_indices == k] for k in range(K)}  

    update_final_mapping()
    
    SINR_k_sets = []

    for k in range(K):
        V_k = Vopt_k_sets[k]
        u_k = Uopt[k, :3]  
        P_k = Uopt[k, 3]   

        distances = np.linalg.norm(V_k[:, :3] - u_k, axis=1)  
        L_r_values = PL_rain(f_c, distances, rain)
        L_s_values = PL_snow(f_c, distances, snow, c)
        L_h_values = PL_hum(f_c, distances, humidity, temper, pressure)
        PL_w = L_r_values + L_s_values + L_h_values 
        fspl = 20 * np.log10(distances) + 20 * np.log10(4 * np.pi * f_c / c)
        h = np.linalg.norm(V_k[:, 2] - u_k[2])
        r = np.linalg.norm(V_k[:, :2] - u_k[:2], axis=1)  
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
            distances_j = np.linalg.norm(V_k[:, :3] - u_j, axis=1)
            L_r_values_j = PL_rain(f_c, distances_j, rain)
            L_s_values_j = PL_snow(f_c, distances_j, snow, c)
            L_h_values_j = PL_hum(f_c, distances_j, humidity, temper, pressure)
            PL_w_j = L_r_values_j + L_s_values_j + L_h_values_j 
            fspl_j = 20 * np.log10(distances_j) + 20 * np.log10(4 * np.pi * f_c / c)
            h_j = np.linalg.norm(V_k[:, 2] - u_j[2])
            r_j = np.linalg.norm(V_k[:, :2] - u_j[:2], axis=1)  
            elevation_angle_j = 180 / np.pi * np.arctan2(h_j, r_j)
            p_LoS_j = 1 / (1 + a * np.exp(-b * (elevation_angle_j - a)))
            p_NLoS_j = 1 - p_LoS_j
            PL_j = fspl_j + p_LoS_j * e_LoS + p_NLoS_j * e_NLoS + PL_w_j
            received_power_j = 10 ** ((P_j - 30 - PL_j) / 10) 
            interf_power_lin += received_power_j

            noise_power_lin = 10 ** ((P_N - 30) / 10)  
            SINR_lin = signal_power_lin / (interf_power_lin + noise_power_lin)
            SINR_dB = 10 * np.log10(SINR_lin)
            SINR_k_sets.append(SINR_dB)
            self.SINR_k_sets = SINR_k_sets  

    progress_dialog.setLabelText("Complete. Please wait...")
    progress_dialog.setRange(0, 0)
    progress_dialog.setValue(0)
    QTimer.singleShot(5000, progress_dialog.close)
    
    self.update_plot(Uopt, Vopt_k_sets)
    self.update_table(Uopt)
    self.update_graph_plots(ee_history, best_ee_history, p_history)
            
    self.reset_button.setEnabled(True)  
    self.generate_button.setEnabled(False)
