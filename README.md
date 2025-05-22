# Energy-efficient deployment simulator of UAV-mounted base stations

A GUI-based **demo-level framework** for unmanned aerial vehicle (UAV) deployment by integrating environmental parameters, radio propagation (path loss) modeling, meteorological data, and terrain data.

This simulator provides an visualization platform for simulating geolocation-aware and weather-aware UAV deployment for energy-efficient network using Python, PyQt6, and scientific computing libraries.


---
## Key Features

- Full manual control of simulation parameters: UAV/UE counts, QoS requirments, frequency, terrain, and region settings
- Real-time weather integration via OpenWeatherMap API, or manual override
- Clustering and optimization-based UAV deployment
- 2D SINR mapping and 3D visualization
- PDF export of UI state
- Modular Python-based architecture for easy customization


---
## Quick start
This simulator has been developed and tested on Windows 10.
```bash
# 1. Clone repository
git clone https://github.com/CNL-sogang/UAV-deployment-simulator.git
cd UAV-deployment-simulator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the simulator
cd build
python mainGUI.py -K [YOUR_API_KEY] -D [CAPTURE_DIRECTORY]
```

> Replace `YOUR_API_KEY` with your OpenWeatherMap API key. (not mandatory.)  
> Replace `CAPTURE_DIRECTORY` with the directory for UI captures. (not mandatory.)  


---
## Command-line options

| Option | Description |
|--------|-------------|
| `-K`   | OpenWeatherMap API key for real-time weather data (optional) |
| `-D`   | Directory to save PDF captures (optional, default: `captures/`) 


---
## Dependencies

- Python 3.9+
- PyQt6
- numpy, matplotlib
- scipy, geopy
- ... etc.

> Full list: [`requirements.txt`](./requirements.txt)

---
## Preview example

<img src="visuals/preview_1.png" alt="Terrain and weather input" height="333">
<img src="visuals/preview_2.png" alt="Multi-UAV visualization" height="350">

---

## Usage Instructions and Recommendations

This simulator was developed for research purposes and is currently provided as a **demo-level framework**. Operators are encouraged to freely modify and extend the code for their own applications.


### Workflow Overview

1. **Parameter input and terrain load**
   - Enter parameters in the input fields.
   - Click **Load** to fetch:
     - Elevation data of the specified region
     - Weather data (real-time or manual)
     - Randomly distributed UEs with varied QoS requirements

2. **Random seed control**
   - Check **Fixed seed** to enable reproducible simulations.

3. **Weather options**
   - Check **Use real-time weather data** and provide a valid **OpenWeatherMap API key** to fetch live weather information.
   - Uncheck the option to **manually input weather parameters**.

4. **Regenerate environment**
   - After clicking Load once, clicking **Load** regenerates a randomized environment with updated UE positions and parameters.

5. **Start simulation**
   - Click **Start** to run the simulation and determine an energy-efficient UAV deployment.
   - Results include:
     - UAV deployment table
     - 2D SINR map
     - 3D terrain plot
     - Algorithm progress graph

6. **Reset**
   - Click **Reset** to return the simulator to its initial state.

---

### UI Tips

- The **graph panel on the right** can be toggled by clicking the hidden blue button labeled **[Weather Conditions]**.

<br>
  <img src="visuals/instruc_2.png" alt="Terrain and weather input" height="350">
<img src="visuals/instruc_1.png" alt="Multi-UAV visualization" height="333">
<br>

- Another hidden blue button labeled **[Simulation Parameters]** can be used to **capture the current UI state as a PDF file**.

---

### External services used

| Data Type | Service | Notes |
|-----------|---------|-------|
| Weather   | [OpenWeatherMap](https://openweathermap.org/) | Requires personal API key |
| Elevation | [elevation-api.eu](https://www.elevation-api.eu) | Free public API |

> ⚠️ These APIs may be subject to usage limits, rate throttling, or unexpected downtime.  

If API reliability is a concern or if customization is required, we strongly recommend modifying the elevation/weather data loading logic in: `core/generate_axes.py`

---
## Future improvements (planned)

- Route planning support
- Base map overlays from tile servers
- ML-based deployment simulation
- Load/save simulation sessions


---
## License

This project is licensed under the **MIT License**.  
© 2025 CNL, Sogang University

---
