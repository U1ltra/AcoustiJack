import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Data
directions = ['R1', 'R2', 'R3', 'R4']
algorithms = ['SORT', 'UCMCTrack', 'SiamRPN', 'DaSiamRPN', 'KCF']
success_rates = [
    0.6551724138, 0.6666666667, 0.7, 0.5833333333,
    0.8, 0.8285714286, 0.8333333333, 0.7777777778,
    0.7666666667, 0.8888888889, 0.7, 0.8,
    0.9310344828, 0.9411764706, 0.8, 0.9090909091,
    0.8, 0.6857142857, 0.8333333333, 0.8055555556
]
dos_values = [
0.9651724138, 0.9066666667, 1, 0.8633333333,
1, 0.9685714286, 1.003333333, 0.9977777778,
0.9666666667, 0.9988888889, 1, 0.97,
1.001034483, 1.001176471, 1, 0.9990909091,
1, 0.9957142857, 1.003333333, 0.9455555556
]

# Reshape success_rates for plotting
success_rates = np.array(success_rates).reshape(5, 4)
# Reshape dos_values for plotting
dos_values = np.array(dos_values).reshape(5, 4)
# Data
data = {
    'SORT': success_rates[0],
    'UCMCTrack': success_rates[1],
    'SiamRPN': success_rates[2],
    'DaSiamRPN': success_rates[3],
    'KCF': success_rates[4],
}

regions = ['D1', 'D2', 'D3', 'D4']
algorithms = list(data.keys())

# Create the bar plot
plt.figure(figsize=(14, 3))

# Set up bar positions
x = np.arange(len(regions)) * 0.8
width = 0.12  # Shorter width for each bar
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
hatches = ['', '///', '...', '+++', 'xxx']  # Different textures for each algorithm

# Create bars for each algorithm
for i, (algorithm, success_rates) in enumerate(data.items()):
    # Main bars (success rates)
    plt.bar(x + i * width, success_rates, width, label=algorithm, 
            color=colors[i], alpha=0.8, hatch=hatches[i], edgecolor='black', linewidth=0.5)
    
    # Extended bars for parentheses values (shallower color)
    plt.bar(x + i * width, dos_values[i], width, 
            color=colors[i], alpha=0.3, edgecolor='black', linewidth=0.5)

# Customize the plot
# plt.xlabel('Victim Object Direction', fontsize=18)
plt.ylabel('Success Rate', fontsize=18)
plt.xlabel('True Target Moving Directions', fontsize=18)

# Add legend for algorithms and extended bars
from matplotlib.lines import Line2D
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.8, 
                                hatch=hatches[i], edgecolor='black', label=alg)
                  for i, alg in enumerate(algorithms)]
legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.3, 
                                   edgecolor='black', label='Disable'))

# plt.legend(handles=legend_elements, fontsize=13, loc='lower left')
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0.0, 1.0)

# Set x-axis labels
plt.xticks(x + width * 2, regions, fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.show()