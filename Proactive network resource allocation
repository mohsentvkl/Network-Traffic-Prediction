# 1- Compare Reactive and Proactive approaches

# %%
def mapping_function2(workload, capacity_range):
    # Define mapping function parameters based on capacity range
    if capacity_range <= 30:
        slope = 1.5
        intercept = 5
    elif 30 < capacity_range <= 40:
        slope = 1.2
        intercept = 8
    elif 40 < capacity_range <= 60:
        slope = 1.1
        intercept = 9
    elif 60 < capacity_range <= 70:
        slope = 1.05
        intercept = 9.5
    elif 70 < capacity_range <= 80:
        slope = 1
        intercept = 10
    else:
        # For capacities above 80, use a default slope and intercept
        slope = 0.9
        intercept = 11

    # Apply the linear mapping function
    required_cores = slope * workload + intercept

    # Ensure that the number of cores is within the specified range [0, capacity_range]
    required_cores = max(0, min(capacity_range, int(required_cores)))
    
    return required_cores


# Example usage:
workload_values = [10, 20, 30]

for workload in workload_values:
    for capacity_range in [30, 50, 80]:
        print(mapping_function2(workload, capacity_range))


# %%
plotdata = pd.DataFrame({
    "LSTM": [31511, 32112, 31278, 31755],
    "GRU" : [30856, 30004, 33000, 31196]
}, index=["FW", "DPI", "IDS", "LB"])

# plot configuration
ax = plotdata.plot(kind="bar", figsize=(12, 5), color=['#4169E1','#FFA500'], width=0.5,fontsize=14) #00CED1, 00BFFF, 1E90FF

# Rotate x-axis labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# set the spacing between the x-axis labels
ax.xaxis.set_tick_params(pad=10) # adjust this value based on your requirements

# Adjust the xlim to add space on the right of the bars
ax.set_xlim(-1, len(plotdata.index))
ax.patch.set_facecolor('white')
#ax.set_facecolor('white')

plt.xlabel("VNF", fontsize=16)
plt.ylabel("MSE value", fontsize=16)

# add legend to the plot
ax.legend(fontsize=14, loc='upper right')

# save the plot
plt.savefig('Comapre_models_accuracy_for_different_VNFS.png', dpi=1080, bbox_inches='tight')

# display the plot
plt.show()

# %% [markdown]
# 

# %% [markdown]
# 2- Copmare MILP and Heuristic strategies

# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming a single cloud with 100 available cores
available_cores_per_cloud = 40
# Initialize RU arrays
ru_reactive = np.zeros(len(real_data_hourly))
ru_proactive = np.zeros(len(predicted_RL_hourly))

# Calculate RU for reactive (real traffic) and proactive (predicted traffic)
for t in range(len(real_data_hourly)):
    # Calculate allocated cores for reactive approach
    allocated_cores_reactive = mapping_function2(real_data_hourly[t], available_cores_per_cloud)
    ru_reactive[t] = allocated_cores_reactive / available_cores_per_cloud
    
    # Calculate allocated cores for proactive approach
    allocated_cores_proactive = mapping_function2(predicted_RL_hourly[t], available_cores_per_cloud)
    ru_proactive[t] = allocated_cores_proactive / available_cores_per_cloud

# Calculate average RU over time for each approach
average_ru_reactive = np.mean(ru_reactive)
average_ru_proactive = np.mean(ru_proactive)

# Calculate reduction percentage in RU
reduction_percentage = (average_ru_reactive - average_ru_proactive) / average_ru_reactive * 100

# Print reduction percentage
print(f"Reduction in Resource Utilization due to Proactive Approach: {reduction_percentage:.2f}%")

# Plot results
plt.figure(figsize=(12, 6))

# Line plot for RU
plt.plot(range(1, 60), ru_reactive[:60], label='Reactive Approach (Real Traffic)', marker='o')
plt.plot(range(1, 60), ru_proactive[:60], label='Proactive Approach (Predicted Traffic)', marker='x')

plt.xlabel('Time (minutes)')
plt.ylabel('Resource Utilization')
plt.title('Comparison of Resource Utilization: Reactive vs Proactive (Single Cloud, One Hour)')
plt.legend()
plt.grid(True)
plt.show()


# %%
# Assuming 5 clouds, each with 100 available cores
num_clouds = 3
available_cores_per_cloud = 80

# Initialize RU arrays
ru_reactive = np.zeros((num_clouds, len(real_data_hourly)))
ru_proactive = np.zeros((num_clouds, len(predicted_RL_hourly)))

# Distribute the traffic across the clouds (e.g., evenly)
def distribute_traffic(traffic, num_clouds):
    return np.array_split(traffic, num_clouds)

# Distribute real and predicted traffic
real_traffic_splits = distribute_traffic(real_data_hourly, num_clouds)
predicted_traffic_splits = distribute_traffic(predicted_RL_hourly, num_clouds)

# Calculate RU for reactive (real traffic) and proactive (predicted traffic)
for cloud_idx in range(num_clouds):
    for t in range(len(real_traffic_splits[cloud_idx])):
        # Calculate allocated cores for reactive approach
        allocated_cores_reactive = mapping_function2(real_traffic_splits[cloud_idx][t], available_cores_per_cloud)
        ru_reactive[cloud_idx, t] = allocated_cores_reactive / available_cores_per_cloud
        
        # Calculate allocated cores for proactive approach
        allocated_cores_proactive = mapping_function2(predicted_traffic_splits[cloud_idx][t], available_cores_per_cloud)
        ru_proactive[cloud_idx, t] = allocated_cores_proactive / available_cores_per_cloud

# Calculate average RU over time for each approach
average_ru_reactive = np.mean(ru_reactive, axis=1)
average_ru_proactive = np.mean(ru_proactive, axis=1)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_clouds + 1), average_ru_reactive, label='Reactive Approach (Real Traffic)', marker='o')
plt.plot(range(1, num_clouds + 1), average_ru_proactive, label='Proactive Approach (Predicted Traffic)', marker='x')

plt.xlabel('Cloud Index')
plt.ylabel('Average Resource Utilization')
plt.title('Comparison of Resource Utilization: Reactive vs Proactive')
plt.xticks(range(1, num_clouds + 1))  # Set x-axis ticks to match cloud indices
plt.legend()
plt.grid(True)
plt.show()


# %%
# Assuming 3 clouds, each with 80 available cores
num_clouds = 5
available_cores_per_cloud = 30

# Initialize RU arrays
ru_reactive = np.zeros((num_clouds, len(real_data)))
ru_proactive = np.zeros((num_clouds, len(predicted_RL)))

# Distribute the traffic across the clouds (e.g., evenly)
def distribute_traffic(traffic, num_clouds):
    return np.array_split(traffic, num_clouds)

# Distribute real and predicted traffic
real_traffic_splits = distribute_traffic(real_data, num_clouds)
predicted_traffic_splits = distribute_traffic(predicted_RL, num_clouds)

# Calculate RU for reactive (real traffic) and proactive (predicted traffic)
for cloud_idx in range(num_clouds):
    for t in range(len(real_traffic_splits[cloud_idx])):
        # Calculate allocated cores for reactive approach
        allocated_cores_reactive = mapping_function2(real_traffic_splits[cloud_idx][t], available_cores_per_cloud)
        ru_reactive[cloud_idx, t] = allocated_cores_reactive / available_cores_per_cloud
        
        # Calculate allocated cores for proactive approach
        allocated_cores_proactive = mapping_function2(predicted_traffic_splits[cloud_idx][t], available_cores_per_cloud)
        ru_proactive[cloud_idx, t] = allocated_cores_proactive / available_cores_per_cloud

# Calculate average RU over time for each approach
average_ru_reactive = np.mean(ru_reactive, axis=1)
average_ru_proactive = np.mean(ru_proactive, axis=1)

# Calculate total average RU across all clouds
average_ru_reactive_total = np.mean(average_ru_reactive)
average_ru_proactive_total = np.mean(average_ru_proactive)

# Calculate the percentage reduction in RU
percentage_reduction = (1 - (average_ru_proactive_total / average_ru_reactive_total)) * 100

# Print results
print(f"Average RU (Reactive Approach): {average_ru_reactive_total:.2f}")
print(f"Average RU (Proactive Approach): {average_ru_proactive_total:.2f}")
print(f"Proactive approach results in a {percentage_reduction:.2f}% reduction in resource utilization compared to the reactive approach.")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_clouds + 1), average_ru_reactive, label='Reactive Approach (Real Traffic)', marker='o')
plt.plot(range(1, num_clouds + 1), average_ru_proactive, label='Proactive Approach (Predicted Traffic)', marker='x')

plt.xlabel('Cloud Index')
plt.ylabel('Average Resource Utilization')
plt.title('Comparison of Resource Utilization: Reactive vs Proactive')
plt.xticks(range(1, num_clouds + 1))  # Set x-axis ticks to match cloud indices
plt.legend()
plt.grid(True)
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# Assuming 5 clouds, each with 100 available cores
num_clouds = 3
available_cores_per_cloud = 80

# Initialize RU arrays
ru_reactive = np.zeros((num_clouds, len(real_data_hourly)))
ru_proactive = np.zeros((num_clouds, len(predicted_RL_hourly)))

# Distribute the traffic across the clouds (dynamically based on capacity)
def distribute_traffic_dynamically(traffic, num_clouds, available_cores_per_cloud):
    cloud_loads = np.zeros(num_clouds)
    allocation = np.zeros((num_clouds, len(traffic)))
    
    for t in range(len(traffic)):
        # Calculate required cores for each cloud
        required_cores = np.zeros(num_clouds)
        for cloud_idx in range(num_clouds):
            required_cores[cloud_idx] = mapping_function2(traffic[t], available_cores_per_cloud)
        
        # Distribute traffic based on available cores
        for cloud_idx in range(num_clouds):
            allocated_cores = min(required_cores[cloud_idx], available_cores_per_cloud - cloud_loads[cloud_idx])
            allocation[cloud_idx, t] = allocated_cores
            cloud_loads[cloud_idx] += allocated_cores
        
        # Reset cloud loads for the next time step
        cloud_loads = np.clip(cloud_loads - allocation[:, t], 0, available_cores_per_cloud)
    
    return allocation

# Distribute real and predicted traffic
real_traffic_splits = distribute_traffic_dynamically(real_data_hourly, num_clouds, available_cores_per_cloud)
predicted_traffic_splits = distribute_traffic_dynamically(predicted_RL_hourly, num_clouds, available_cores_per_cloud)

# Calculate RU for reactive (real traffic) and proactive (predicted traffic)
for cloud_idx in range(num_clouds):
    for t in range(len(real_data_hourly)):
        allocated_cores_reactive = real_traffic_splits[cloud_idx, t]
        ru_reactive[cloud_idx, t] = allocated_cores_reactive / available_cores_per_cloud
        
        allocated_cores_proactive = predicted_traffic_splits[cloud_idx, t]
        ru_proactive[cloud_idx, t] = allocated_cores_proactive / available_cores_per_cloud

# Calculate average RU over time for each approach
average_ru_reactive = np.mean(ru_reactive, axis=1)
average_ru_proactive = np.mean(ru_proactive, axis=1)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(range(1, num_clouds + 1), average_ru_reactive, label='Reactive Approach (Real Traffic)', marker='o')
plt.plot(range(1, num_clouds + 1), average_ru_proactive, label='Proactive Approach (Predicted Traffic)', marker='x')

plt.xlabel('Cloud Index')
plt.ylabel('Average Resource Utilization')
plt.title('Comparison of Resource Utilization: Reactive vs Proactive')
plt.xticks(range(1, num_clouds + 1))  # Set x-axis ticks to match cloud indices
plt.legend()
plt.grid(True)
plt.show()


# %%
# Plot the results

## First plot: Effect of increasing number of clouds
plt.figure(figsize=(12, 8))
clouds = list(cloud_configurations.keys())
milp_avg_times_clouds = [np.mean(milp_times[num_clouds]) for num_clouds in clouds]
heuristic_avg_times_clouds = [np.mean(heuristic_times[num_clouds]) for num_clouds in clouds]

plt.plot(clouds, milp_avg_times_clouds, label='MILP Average Times')
plt.plot(clouds, heuristic_avg_times_clouds, label='Heuristic Average Times')

plt.xlabel('Number of Clouds')
plt.ylabel('Average Execution Time (seconds)')
plt.title('Effect of Increasing Number of Clouds on Execution Times')
plt.legend()
plt.grid(True)

# Set the range of the x-axis
plt.xlim(min(clouds), max(clouds))  # Adjust as needed

plt.savefig('effect_of_clouds.png', dpi=1080, bbox_inches='tight')
plt.show()


# Second plot: Effect of increasing time intervals
plt.figure(figsize=(12, 8))
for num_clouds in cloud_configurations.keys():
    plt.plot(problem_sizes, milp_times[num_clouds], label=f'MILP - {num_clouds} Clouds')
    plt.plot(problem_sizes, heuristic_times[num_clouds], label=f'Heuristic - {num_clouds} Clouds')

plt.xlabel('Problem Size (Number of Time Intervals)')
plt.ylabel('Execution Time (seconds)')
plt.title('Effect of Increasing Time Intervals on Execution Times')
plt.legend()
plt.grid(True)
plt.xlim(min(problem_sizes), max(problem_sizes)) 
plt.savefig('effect_of_time_intervals.png', dpi=1080, bbox_inches='tight')
plt.show()

# Combined plot: Effect of both intervals and clouds
fig, ax = plt.subplots(figsize=(12, 8))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

for idx, num_clouds in enumerate(cloud_configurations.keys()):
    ax.plot(problem_sizes, milp_times[num_clouds], color=colors[idx % len(colors)], linestyle='-', label=f'MILP - {num_clouds} Clouds')
    ax.plot(problem_sizes, heuristic_times[num_clouds], color=colors[idx % len(colors)], linestyle='--', label=f'Heuristic - {num_clouds} Clouds')

ax.set_xlabel('Problem Size (Number of Time Intervals)')
ax.set_ylabel('Execution Time (seconds)')
ax.set_title('Effect of Both Intervals and Clouds on Execution Times')
ax.legend()
ax.grid(True)
plt.savefig('combined_effects.png', dpi=1080, bbox_inches='tight')
plt.show()
