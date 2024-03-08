import matplotlib.pyplot as plt

def update_plot(results, save_path=None):
    x, y1, y2 = zip(*results)
    plt.plot(x, y1, color='r', label='first')
    plt.plot(x, y2, color='b', label='second')
    plt.title('Agent win rate against Random Agent')
    plt.xlabel('episodes')
    plt.ylabel('win rate')
    plt.legend() 
    plt.grid(True)
    if save_path: plt.savefig(save_path)
    plt.show()