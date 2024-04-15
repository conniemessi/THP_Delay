import numpy as np
from scipy.stats import poisson
import random
from scipy.integrate import quad


def g_function(t):
    # Define the impulse response function g_uu'
    return np.exp(-t ** 2)


def simulate_hawkes_thinning(baseline_intensity, triggering_intensity, end_time, delay_matrix):
    """Simulates a Hawkes process using thinning method.

  Args:
      baseline_intensity: Function defining baseline intensity for each dimension.
      triggering_intensity: Function calculating intensity based on past events.
      end_time: Maximum simulation time.

  Returns:
      A list of lists containing timestamps and event types (dimensions) for accepted events.
  """

    # Simulate Poisson process for each dimension (get initial timestamps)
    timestamps_all = []
    for dim in range(num_dimensions - 1):
        timestamps_all.append(generate_poisson_events(baseline_intensity[dim], end_time)[1])
        # timestamps_all.append(simulate_poisson_process(baseline_intensity(dim), end_time))
    # Thinning based on intensity
    accepted_events = []
    for dim, timestamps in enumerate(timestamps_all):
        for timestamp in timestamps:
            intensity = baseline_intensity[dim]
            # if t == (num_dimensions - 1):
            #     print(t)
            #     total_intensity = calculate_intensity(t, timestamp, timestamps_all, triggering_intensity)
            if random.random() <= intensity:
                accepted_events.append([timestamp, dim])  # Store timestamp and dimension

            intensity = sum((triggering_intensity[u_prime] *
                         quad(lambda s: g_function(timestamp - delay_matrix[u_prime] - s), 0, timestamp - delay_matrix[u_prime])[0] \
                         for u_prime in range(num_dimensions)))
            if (random.random() <= intensity) and (timestamp + delay_matrix[dim] < end_time):
                accepted_events.append([timestamp + delay_matrix[dim], num_dimensions - 1])

    return accepted_events


# def simulate_poisson_process(lambda_, end_time):
#     """Simulates a Poisson process with specific lambda (intensity).
#
#   Args:
#       lambda_: Mean arrival rate for the Poisson process.
#       end_time: Maximum simulation time.
#
#   Returns:
#       A list of timestamps for the simulated Poisson process.
#   """
#
#     # Use inverse transform sampling for efficiency
#     u = np.random.rand(int(-np.log(1 - np.random.rand()) / lambda_))
#     timestamps = u / lambda_
#     timestamps = timestamps[timestamps <= end_time]
#     return timestamps.tolist()


def generate_poisson_events(rate, time_duration):
    num_events = np.random.poisson(rate * time_duration)
    event_times = np.sort(np.random.uniform(0, time_duration, num_events))
    inter_arrival_times = np.diff(event_times)
    return num_events, event_times, inter_arrival_times


# # Define functions for calculating intensity based on past events (replace with your specific logic)
# def calculate_intensity(dim, t, timestamps_all, triggering_intensity):
#     """Calculates the total intensity at time t based on past events and baseline intensity.
#
#   This function needs to be implemented based on your specific Hawkes process setup.
#
#   Args:
#       t: Current time.
#       timestamps_all: List of lists containing timestamps for all dimensions.
#       triggering_intensity: Function defining the triggering intensity based on past events.
#
#   Returns:
#       The total intensity at time t.
#   """
#
#     total_intensity = 0.0
#     for timestamps in timestamps_all:
#         for timestamp in timestamps:
#             if t - timestamp < 1.0:  # Consider events within a decay window (adjust as needed)
#                 # total_intensity += triggering_intensity(t - timestamp)  # Implement your triggering function
#                 total_intensity += triggering_intensity[dim] * np.exp(t - timestamp)
#     total_intensity += baseline_intensity[dim]  # Add baseline intensity
#     return total_intensity


# Define parameters
num_dimensions = 5
baseline_intensity = np.array([2, 1.5, 1, 0.5, 0])
triggering_intensity = np.array([2, 1.5, 1, 0.5, 0])
delay_matrix = np.array([2, 1.5, 1, 0.5, 0])
end_time = 20.0

# Run simulation and collect results
all_events = []
num_seq = 10
for _ in range(num_seq):  # Run simulation 1000 times
    accepted_events = simulate_hawkes_thinning(baseline_intensity, triggering_intensity, end_time, delay_matrix)
    accepted_events = sorted(accepted_events, key=lambda x: x[0])
    all_events.append(accepted_events)

# Print results
print(all_events[0])

# save as txt
# event time
with open('events_time.txt', 'w') as file:
    for seq_id in range(num_seq):
        # 'item + 1' indicate that event type start from 1
        for time, event in all_events[seq_id]:
            file.write(str(time))
            file.write(' ')
        file.write('\n')

# event type
with open('events_type.txt', 'w') as file:
    for seq_id in range(num_seq):
        # 'item + 1' indicate that event type start from 1
        for time, event in all_events[seq_id]:
            file.write(str(event + 1))
            file.write(' ')
        file.write('\n')