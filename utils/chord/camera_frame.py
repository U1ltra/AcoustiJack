import matplotlib.pyplot as plt
import numpy as np

# Data
frame_rate = ['30Hz', '60Hz']
algorithms = ['SORT', 'UCMCTrack', 'SiamRPN', 'DaSiamRPN', 'KCF']
success_rates = [
    85.2, 69.2,
    77.8, 63.6,
    81.5, 79.6,
    92.6, 83.3,
    77.8, 46.3,
]
parentheses_values = [
    100, 94.2,
    96.3, 90.9,
    92.6, 87.0,
    98.1, 94.4,
    98.1, 94.4,
]

# Reshape data for plotting
success_rates = np.array(success_rates).reshape(5, 2) / 100  # Convert to 0-1 scale
parentheses_values = np.array(parentheses_values).reshape(5, 2) / 100  # Convert to 0-1 scale

# Create the bar plot
plt.figure(figsize=(5, 3))

# Set up bar positions
x = np.arange(len(frame_rate)) * 0.8
width = 0.12  # Same width as other plots
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Match algorithm order
hatches = ['', '///', '...', '+++', 'xxx']  # Different textures for each algorithm

# Create bars for each algorithm
for i, algorithm in enumerate(algorithms):
    # Main bars (success rates)
    plt.bar(x + i * width, success_rates[i], width, label=algorithm, 
            color=colors[i], alpha=0.8, hatch=hatches[i], edgecolor='black', linewidth=0.5)
    
    # Extended bars for parentheses values (shallower color)
    plt.bar(x + i * width, parentheses_values[i], width, 
            color=colors[i], alpha=0.3, edgecolor='black', linewidth=0.5)

# Customize the plot
plt.xlabel('Camera Frame Rate', fontsize=18)
plt.ylabel('Success Rate', fontsize=18)

# Add legend for algorithms and extended bars
from matplotlib.lines import Line2D
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.8, 
                                hatch=hatches[i], edgecolor='black', label=alg)
                  for i, alg in enumerate(algorithms)]
legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.3, 
                                   edgecolor='black', label='Extended'))

# plt.legend(handles=legend_elements, fontsize=13, loc='lower left')
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0.0, 1.05)

# Set x-axis labels
plt.xticks(x + width * 2, frame_rate, fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.show()