import matplotlib.pyplot as plt
import numpy as np

# Data
heights = ['5m', '10m', '15m', '20m']
algorithms = [ 'SORT', 'UCMCTrack', 'SiamRPN', 'DaSiamRPN', 'KCF']
success_rates = [
    81.5, 85.2, 24.1, 3.7,
    70.4, 77.8, 72.2, 25.9,
    79.6, 81.5, 87.0, 55.6,
    90.7, 92.6, 81.5, 79.6,
    72.2, 77.8, 44.4, 35.2,
]
parentheses_values = [
    100, 100, 100, 100,
    92.6, 96.3, 100, 98.1,
    92.6, 92.6, 94.4, 100,
    92.6, 98.1, 90.7, 98.1,
    100, 98.1, 100, 100,
]

# Reshape data for plotting
success_rates = np.array(success_rates).reshape(5, 4) / 100  # Convert to 0-1 scale
parentheses_values = np.array(parentheses_values).reshape(5, 4) / 100  # Convert to 0-1 scale

# Create the bar plot
plt.figure(figsize=(14, 3))

# Set up bar positions
x = np.arange(len(heights)) * 0.8
width = 0.12  # Same width as your other plots
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Same order as algorithms
hatches = ['', '///', '...', '+++', 'xxx']  # Different textures for each algorithm

# Create bars for each algorithm
for i, algorithm in enumerate(algorithms):
    # Main bars (success rates)
    plt.bar(x + i * width, success_rates[i], width, label=algorithm, 
            color=colors[i], alpha=0.8, hatch=hatches[i], edgecolor='black', linewidth=0.5)
    
    # Extended bars for Disable values (shallower color)
    plt.bar(x + i * width, parentheses_values[i], width, 
            color=colors[i], alpha=0.3, edgecolor='black', linewidth=0.5)

# Customize the plot
plt.xlabel('UAV Height', fontsize=18)
plt.ylabel('Success Rate', fontsize=18)
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0.0, 1.05)

# Set x-axis labels
plt.xticks(x + width * 2, heights, fontsize=16)
plt.yticks(fontsize=16)

# Add legend for algorithms and extended bars
from matplotlib.lines import Line2D
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.8, 
                                hatch=hatches[i], edgecolor='black', label=alg)
                  for i, alg in enumerate(algorithms)]
legend_elements.insert(0, plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.3, 
                                   edgecolor='black', label='Disable'))

# plt.legend(handles=legend_elements, fontsize=13, loc='lower left')

plt.tight_layout()
plt.show()