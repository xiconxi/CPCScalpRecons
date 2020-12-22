import matplotlib.pyplot as plt
import numpy as np 
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
sizes = np.ones(12*2)
labels = [str(i*15) for i in range(-11, 13)]
fig1, ax1 = plt.subplots()

theme = plt.get_cmap('hsv')
ax1.set_prop_cycle("color", [theme(1. * i / len(sizes)) for i in range(len(sizes))])

ax1.pie(sizes, startangle=90, labels=labels)
plt.pie(sizes, radius=0.6,colors = 'w')

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()