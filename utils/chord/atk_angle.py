import matplotlib.pyplot as plt
import numpy as np

# Data
degrees = ['0°', '30°', '60°', '90°']
algorithms = [ 'SORT', 'SiamRPN']
success_rates = [
0.75, 0.6129032258, 0.59375, 0.09090909091,
0.9444444444, 0.9393939394, 0.5757575758, 0.2121212121,
]
parentheses_values = [
1, 1, 1, 1,
1, 1, 0.8181818182, 0.5151515152,
]


# Reshape data for plotting
success_rates = np.array(success_rates).reshape(2, 4)
parentheses_values = np.array(parentheses_values).reshape(2, 4)

# Create the line plot
plt.figure(figsize=(4, 5))

# Colors and markers matching the established style
colors = ['#ff7f0e', '#d62728']
markers = ['o', 's']  # Different markers for each algorithm
linestyles = ['-', '--']  # Different line styles

# Create lines for each algorithm
for i, algorithm in enumerate(algorithms):
    plt.plot(degrees, success_rates[i], marker=markers[i], label=algorithm,
            color=colors[i], linewidth=2, markersize=8, linestyle=linestyles[i])

# Customize the plot
plt.xlabel('False Target Relative Angle', fontsize=16)
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
