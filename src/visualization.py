import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def plot_lattice(lattice, step, output_path=None):
    plt.figure(figsize=(6, 6))
    plt.imshow(lattice, cmap='coolwarm', interpolation='bicubic')
    plt.title(f"Ising Model - Step {step}")
    plt.colorbar(label="Spin State")
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()


def create_interactive_plot(filename):
    data = np.load(filename)
    frames = [data[f] for f in data.files]
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    
    img = ax.imshow(frames[0], cmap='coolwarm', interpolation='bicubic')
    
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Step', 0, len(frames)-1, valinit=0, valfmt='%0.0f')

    def update(val):
        idx = int(slider.val)
        img.set_data(frames[idx])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()