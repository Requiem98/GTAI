import matplotlib.colors as mcolors

fig = plt.figure(figsize=(9,6), dpi=500)	#identifies the figure 
plt.title("Steering angle Predictions VS Target", fontsize='18', fontweight="bold")	#title

a=1000
y_pred = bf.reverse_normalized_steering(o[1:a])
y_true = data["steeringAngle"][:a]
x = np.arange(a)

plt.plot(x, y_true, color=mcolors.CSS4_COLORS["mediumseagreen"])
plt.plot(y_pred, color=mcolors.CSS4_COLORS["black"])
plt.xlabel("I-th prediction",fontsize='13')	#adds a label in the x axis
plt.ylabel("Steering Angle",fontsize='13')	#adds a label in the y axis
plt.legend(("Target Angle", 'Predicted Angle'), loc='best')	#creates a legend to identify the plot
plt.savefig('angle_pred_vs_true.png')	#saves the figure in the present directory
plt.grid(axis='both', alpha=.3)

plt.gca().spines["top"].set_alpha(0.0)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.show()