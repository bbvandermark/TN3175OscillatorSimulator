import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def create_animation(t_vals, x_vals, filename):
    fig = plt.figure()
    ax = plt.axes(xlim=(-5, 5), ylim=(-1, 1))

    width = 1 / len(x_vals) * 2
    height = width / 5

    def update(frame):
        ax.cla()
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'pink', 'brown', 'gray']
        for i in range(len(x_vals)):
            ax.add_patch(plt.Rectangle((x_vals[i][frame] - 0.5, 0), width, height, color=colors[i]))

        plt.xlim(-5, 5)
        plt.ylim(-1, 1)
        plt.xlabel("x")
        plt.title("t = {:.2f}".format(t_vals[frame]))
        return fig

    frame_count = len(t_vals)
    anim = animation.FuncAnimation(fig, update, frames=frame_count, interval=100, repeat=False)
    anim.save(filename, fps=30)
    plt.clf()
    print(f"animation saved to {filename}")