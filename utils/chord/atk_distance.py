import matplotlib.pyplot as plt
import numpy as np

# Data
distance = ['1m', '2m', '3m']
algorithms = ['SORT', 'SiamRPN']
success_rates = [
    81.5, 71.4, 71.7,
    85.2, 43.9, 50.0,
]

# Reshape data for plotting
success_rates = np.array(success_rates).reshape(2, 3) / 100  # Convert to 0-1 scale

# Create the line plot
plt.figure(figsize=(4, 5))

# Colors and markers matching the established style
colors = ['#ff7f0e', '#d62728']  # SiamRPN (orange) and SORT (red) from your color scheme
markers = ['o', 's']  # Different markers for each algorithm
linestyles = ['-', '--']  # Different line styles

# Create lines for each algorithm
for i, algorithm in enumerate(algorithms):
    plt.plot(distance, success_rates[i], marker=markers[i], label=algorithm,
            color=colors[i], linewidth=2, markersize=8, linestyle=linestyles[i])

# Customize the plot
plt.xlabel('False Target Distance', fontsize=18)
plt.ylabel('Success Rate', fontsize=18)
plt.grid(True, alpha=0.3)
plt.ylim(0.0, 1.0)

# Set font sizes
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Add legend
# plt.legend(fontsize=13, loc='lower left')

plt.tight_layout()
plt.show()