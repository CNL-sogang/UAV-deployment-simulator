def update_graph_plots(self, ee_history, best_ee_history, p_history):
    for i, (figure, canvas) in enumerate(zip(self.figures_3, self.canvases_3)):
        figure.clear()
        if i == 0:
            ax = figure.add_subplot(111)
            ax.plot(range(len(best_ee_history)), best_ee_history, marker='none', linestyle='-')
            ax.plot(len(best_ee_history)-1, best_ee_history[-1], marker='*', markersize=10, markerfacecolor='red', color='red', linestyle='none')
            ax.plot(range(len(ee_history)), ee_history, marker='none', linestyle='--')
            ax.set_title("Energy Efficiency (bits/Joule)\n", fontweight='bold', fontsize=11)
            ax.set_xlabel("Iteration Progress", fontsize=9)
            ax.minorticks_on()
            ax.tick_params(which='both', direction='in', top=True, right=True, left=True, bottom=True)
            figure.tight_layout()
        else:
            ax1 = figure.add_subplot(111)
            ax1.plot(range(len(p_history)), p_history, marker='none', linestyle='-')
            ax1.plot(len(p_history)-1, p_history[-1], marker='*', markersize=10, markerfacecolor='red', color='red', linestyle='none')
            ax1.set_title("Total Transmit Power (dBm)\n", fontweight='bold', fontsize=11)
            ax1.set_xlabel("Iteration Progress", fontsize=9)
            ax1.minorticks_on()
            ax1.tick_params(which='both', direction='in', top=True, right=True, left=True, bottom=True)
            figure.tight_layout()
        canvas.draw()
    