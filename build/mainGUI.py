import sys
import os
from PyQt6.QtWidgets import QApplication, QWidget
import argparse

base_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(base_path, '..', 'build')))
sys.path.append(os.path.abspath(os.path.join(base_path, '..', 'core')))
sys.path.append(os.path.abspath(os.path.join(base_path, '..', 'plotting')))
sys.path.append(os.path.abspath(os.path.join(base_path, '..', 'simulation')))
sys.path.append(os.path.abspath(os.path.join(base_path, '..', 'utils')))

from initUI import initUI
from clear_results import clear_results
from reset_to_initial import reset_to_initial
from generate_axes import generate_axes
from run_simulation import run_simulation
from update_plot import update_plot
from update_table import update_table
from update_weather import update_weather
from update_graph_plots import update_graph_plots
from toggle_random_seed_input import toggle_random_seed_input
from weather_input import weather_input
from toggle_region3 import toggle_region3
from capture_and_save import capture_and_save

def parse_args():
    parser = argparse.ArgumentParser(description="UAV Deployment Simulator")
    parser.add_argument('-K', type=str, help='OpenWeatherMap API key; If no argument, it can be simulated by manual input of weather conditions.')
    parser.add_argument('-D', type=str, default='captures', help='PDF capture save directory, default is /captures')
    return parser.parse_args()

class SimulatorApp(QWidget):
    def __init__(self, api_key=None, save_dir=None):
        super().__init__()
        self.simulation_params = {}  
        self.hyperparams = {}
        self.api_key = api_key
        self.capture_save_dir = save_dir or "captures"
        self.initUI = initUI.__get__(self)
        self.clear_results = clear_results.__get__(self)
        self.reset_to_initial = reset_to_initial.__get__(self)
        self.generate_axes = generate_axes.__get__(self)
        self.run_simulation = run_simulation.__get__(self)
        self.update_plot = update_plot.__get__(self)
        self.update_table = update_table.__get__(self)
        self.update_weather = update_weather.__get__(self)
        self.update_graph_plots = update_graph_plots.__get__(self)
        self.toggle_random_seed_input = toggle_random_seed_input.__get__(self)
        self.weather_input = weather_input.__get__(self)
        self.toggle_region3 = toggle_region3.__get__(self)
        self.capture_and_save = capture_and_save.__get__(self)
        self.initUI()
        
if __name__:
    args = parse_args()
    app = QApplication(sys.argv)
    simulator = SimulatorApp(api_key=args.K, save_dir=os.path.abspath(args.D))
    sys.exit(app.exec())