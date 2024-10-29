import streamlit as st
import numpy as np


def app():
    st.header('Airport Runway Capacity')

    with (st.form(key='runway_capacity')):
        # Input columns for aircraft data
        col1, col2, col3 = st.columns(3)

        # Aircraft A inputs
        with col1:
            st.subheader('Aircraft A')
            approach_speed_A = st.text_input("Approach Speed (km/h)", key='approach_speed_A')
            mix_A = st.text_input("Mix (%)", key='mix_A')
            ri_A = st.text_input("Ri (s)", key='ri_A')

        # Aircraft B inputs
        with col2:
            st.subheader('Aircraft B')
            approach_speed_B = st.text_input("Approach Speed (km/h)", key='approach_speed_B')
            mix_B = st.text_input("Mix (%)", key='mix_B')
            ri_B = st.text_input("Ri (s)", key='ri_B')

        # Aircraft C inputs
        with col3:
            st.subheader('Aircraft C')
            approach_speed_C = st.text_input("Approach Speed (km/h)", key='approach_speed_C')
            mix_C = st.text_input("Mix (%)", key='mix_C')
            ri_C = st.text_input("Ri (s)", key='ri_C')

        # Separation matrix inputs
        st.subheader('Minimum Separation Matrix')
        st.write('Enter the minimum separation (km) between the aircrafts in approach airspace')

        # Creating a 3x3 matrix for minimum separation distances
        separation_matrix = []
        separation_cols = st.columns(3)

        for i, aircraft_trail in enumerate(['A', 'B', 'C']):
            row = []
            for j, aircraft_lead in enumerate(['A', 'B', 'C']):
                value = separation_cols[j].text_input(
                    f"{aircraft_lead} Leading, {aircraft_trail} Trailing", key=f'sep_{aircraft_lead}_{aircraft_trail}')
                row.append(value)
            separation_matrix.append(row)

        st.subheader('Additional Parameters')
        approach_path_length = st.text_input("Length of common approach path (km)", key='approach_path_length')
        min_distance_from_threshold = st.text_input(
            "Minimum distance of arriving aircraft from threshold to release of a departure (km)",
            key='min_distance_from_threshold')
        departure_service_time = st.text_input("Departure service time (s)", key='departure_service_time')

        submitButton = st.form_submit_button(label='Calculate')

        if submitButton:

            try:
                # Convert inputs to float where necessary
                speeds = [float(approach_speed_A), float(approach_speed_B), float(approach_speed_C)]  # Speeds in km/h
                gamma = float(approach_path_length)  # The common approach path length

                # Convert separation matrix values to float
                separation_matrix_float = [[float(value) for value in row] for row in separation_matrix]

                # Transpose the separation matrix
                separation_matrix_transposed = np.transpose(separation_matrix_float)

                # Function to calculate Tij
                def calculate_Tij(vi, vj, delta_ij, gamma):
                    if vi <= vj:
                        # Condition (i): Vi <= Vj
                        return delta_ij / vj
                    else:
                        # Condition (ii): Vi > Vj
                        return (delta_ij / vi) + gamma * ((1 / vj) - (1 / vi))

                # Create Tij matrix in hours
                Tij_matrix = np.zeros((3, 3))
                aircraft_types = ['A', 'B', 'C']

                for i in range(3):
                    for j in range(3):
                        Tij_matrix[i][j] = calculate_Tij(
                            speeds[i],  # vi - trailing aircraft speed
                            speeds[j],  # vj - leading aircraft speed
                            separation_matrix_transposed[i][j],  # delta_ij - separation from matrix
                            gamma  # gamma - common approach path length
                        )

                # Convert Tij from hours to seconds
                Tij_matrix_seconds = Tij_matrix * 3600

                # Now, calculate and display Pij matrix
                # Convert Mix inputs to probabilities
                mix_values = [
                    float(mix_A) / 100,  # Mix of A
                    float(mix_B) / 100,  # Mix of B
                    float(mix_C) / 100  # Mix of C
                ]

                # Display Tij matrix in seconds
                Tij_matrix_transposed = Tij_matrix_seconds.T

                # Calculate Pij matrix
                Pij_matrix = np.zeros((3, 3))
                for i in range(3):
                    for j in range(3):
                        Pij_matrix[i][j] = mix_values[i] * mix_values[j]  # Pi_j = (mix of trailing)*(mix of leading)

                # Calculate Eij as the sum of all element-wise products of Tij * Pij
                Eij_value = np.sum(Tij_matrix_transposed * Pij_matrix)

                capacity = int(np.round(3600/Eij_value))

                ri_values = [
                    float(ri_A),  # Ri of A
                    float(ri_B),  # Ri of B
                    float(ri_C)  # Ri of C
                ]

                E_Ri = mix_values[0] * ri_values[0] + mix_values[1] * ri_values[1] + mix_values[2] * ri_values[2]

                # Calculate E[δd/vj]
                E_δd_vj = ((mix_values[0] / speeds[0]) + (mix_values[1] / speeds[1]) + (mix_values[2] / speeds[2])
                             ) * float(min_distance_from_threshold) * 3600  # Convert to seconds

                # Calculate the sum of E[Ri] and E[δd/vj]
                sum_E_Ri_E_δd_vj = E_Ri + E_δd_vj

                departure_service_time_float = float(departure_service_time)  # Convert departure service time to float

                # Display matrices side by side
                col1, col2 = st.columns(2)  # Create two columns for the matrices

                # Display Tij matrix in first column
                with col1:
                    st.write("[Tij](in s):")
                    st.write(Tij_matrix_transposed)

                # Display Pij matrix in second column
                with col2:
                    st.write("[Pij]:")
                    st.write(Pij_matrix)

                # Display Eij value
                st.write(f"E[Tij] = {Eij_value} s")

                st.subheader(f"Capacity (Arrivals Only) = {capacity} arrivals/hour")

                # Display E[Ri] and E[δd/vj]
                st.write(f"E[Ri] = {E_Ri} s")
                st.write(f"E[δd/Vj] = {E_δd_vj} s")

                # Display the comparison result
                st.write(f"E[Tij] ≥ {sum_E_Ri_E_δd_vj} + (n-1) × {departure_service_time_float}")

                n = 1
                max_Tij_value = np.max(Tij_matrix_transposed)  # The largest value from the Tij matrix
                current_E_Tij = Eij_value  # Start with the calculated E[Tij]

                last_E_Tij = None  # Placeholder for the last E[Tij] lower bound

                while current_E_Tij < max_Tij_value:
                    # Calculate the next E[Tij] value for release of 'n' departures
                    next_E_Tij1 = sum_E_Ri_E_δd_vj + (n - 1) * departure_service_time_float
                    next_E_Tij2 = sum_E_Ri_E_δd_vj + n * departure_service_time_float

                    # Print the statement for the current n
                    st.write(f"To release {n} departure(s), {next_E_Tij1} ≤ E[Tij] < {next_E_Tij2}")

                    # Store the current upper bound for use in the final statement
                    last_E_Tij = next_E_Tij2

                    # Increment n and update the E[Tij] for the next iteration
                    n += 1
                    current_E_Tij = next_E_Tij2  # Update to the next upper bound (not the lower bound)

                # For the last condition where E[Tij] is greater than or equal to the largest value
                st.write(f"To release {n} departure(s), E[Tij] ≥ {last_E_Tij}")

                # Use the calculated minimum threshold from sum_E_Ri_E_δd_vj
                min_threshold = sum_E_Ri_E_δd_vj  # This will change based on the input values

                # Create the departure_matrix based on Tij_matrix_transposed values
                departure_matrix = np.zeros_like(Tij_matrix_transposed)

                # Calculate the upper and lower bounds for each departure range
                n = 1
                departure_ranges = []
                while True:
                    lower_bound = sum_E_Ri_E_δd_vj + (n - 1) * departure_service_time_float
                    upper_bound = sum_E_Ri_E_δd_vj + n * departure_service_time_float
                    departure_ranges.append((lower_bound, upper_bound))

                    # Stop when the upper bound exceeds the maximum value in Tij_matrix_transposed
                    if upper_bound >= np.max(Tij_matrix_transposed):
                        break
                    n += 1

                # Now, map each value in Tij_matrix_transposed to the correct departure count
                for i in range(3):
                    for j in range(3):
                        Tij_value = Tij_matrix_transposed[i][j]

                        # If Tij value is below the calculated minimum threshold, set it to 0
                        if Tij_value < min_threshold:
                            departure_matrix[i][j] = 0
                        else:
                            # Map the Tij value to the correct departure count
                            for n, (lower_bound, upper_bound) in enumerate(departure_ranges, start=1):
                                if lower_bound <= Tij_value < upper_bound:
                                    departure_matrix[i][j] = n
                                    break
                            else:
                                # If the value is greater than the last range, assign the highest departure count
                                departure_matrix[i][j] = n + 1

                # Display the departure_matrix
                st.write("Departure Matrix:")
                st.write(departure_matrix)

                # Dictionary to store the sum of Pij values for each number of departures
                departure_probabilities = {}

                # Loop through the departures matrix and map it to the Pij matrix
                for i in range(len(departure_matrix)):
                    for j in range(len(departure_matrix[i])):
                        # Get the number of departures and the corresponding Pij value
                        num_departures = int(departure_matrix[i][j])
                        pij_value = Pij_matrix[i][j]

                        # Add the Pij value to the appropriate departures bucket
                        if num_departures > 0:
                            if num_departures in departure_probabilities:
                                departure_probabilities[num_departures] += pij_value
                            else:
                                departure_probabilities[num_departures] = pij_value

                # Print the total probability for each number of departures
                for num_departures, total_probability in departure_probabilities.items():
                    st.write(f"Probability to release {num_departures} departure(s) = {total_probability:.4f}")

                # Calculate the mixed operation capacity
                sum_prob_n_times_n = 0

                # Loop through the dictionary to calculate the sum of (prob of n departures * n)
                for num_departures, total_probability in departure_probabilities.items():
                    sum_prob_n_times_n += total_probability * num_departures

                # Calculate the capacity (Mixed Operation)
                capacity_mixed_operation = int(np.round((3600 * (1 + sum_prob_n_times_n)) / Eij_value))

                # Display the calculated mixed operation capacity
                st.subheader(f"Capacity (Mixed Operation) = {capacity_mixed_operation: } operations/hour")

                # Check if any value in Tij_matrix_transposed is less than min_threshold
                if np.any(Tij_matrix_transposed < min_threshold):
                    st.write("To release at least one departure between each pair of arrivals")

                    # Replace values in Tij_matrix_transposed that are less than min_threshold with min_threshold
                    Tij_matrix_transposed_modified = np.where(Tij_matrix_transposed < min_threshold, min_threshold,
                                                              Tij_matrix_transposed)

                    # Print the modified Tij_matrix_transposed
                    st.write("Modified [Tij](in s):")
                    st.write(Tij_matrix_transposed_modified)

                    # Recalculate the probability to release 1 departure using the modified Tij matrix
                    departure_matrix_modified = np.zeros_like(Tij_matrix_transposed_modified)

                    # Calculate the upper and lower bounds for each departure range
                    n = 1
                    departure_ranges = []
                    while True:
                        lower_bound = sum_E_Ri_E_δd_vj + (n - 1) * departure_service_time_float
                        upper_bound = sum_E_Ri_E_δd_vj + n * departure_service_time_float
                        departure_ranges.append((lower_bound, upper_bound))

                        # Stop when the upper bound exceeds the maximum value in Tij_matrix_transposed_modified
                        if upper_bound >= np.max(Tij_matrix_transposed_modified):
                            break
                        n += 1

                    # Now, map each value in the modified Tij_matrix_transposed to the correct departure count
                    for i in range(3):
                        for j in range(3):
                            Tij_value = Tij_matrix_transposed_modified[i][j]

                            # If Tij value is below the calculated minimum threshold, set it to 0
                            if Tij_value < min_threshold:
                                departure_matrix_modified[i][j] = 0
                            else:
                                # Map the Tij value to the correct departure count
                                for n, (lower_bound, upper_bound) in enumerate(departure_ranges, start=1):
                                    if lower_bound <= Tij_value < upper_bound:
                                        departure_matrix_modified[i][j] = n
                                        break
                                else:
                                    # If the value is greater than the last range, assign the highest departure count
                                    departure_matrix_modified[i][j] = n + 1

                    # Dictionary to store the sum of Pij values for each number of departures for the modified matrix
                    departure_probabilities_modified = {}

                    # Loop through the modified departures matrix and map it to the Pij matrix
                    for i in range(len(departure_matrix_modified)):
                        for j in range(len(departure_matrix_modified[i])):
                            # Get the number of departures and the corresponding Pij value
                            num_departures = int(departure_matrix_modified[i][j])
                            pij_value = Pij_matrix[i][j]

                            # Add the Pij value to the appropriate departures bucket
                            if num_departures > 0:
                                if num_departures in departure_probabilities_modified:
                                    departure_probabilities_modified[num_departures] += pij_value
                                else:
                                    departure_probabilities_modified[num_departures] = pij_value

                    # Print the new probability to release 1 departure
                    prob_one_departure_modified = departure_probabilities_modified.get(1)
                    st.write(f"New Probability to release 1 departure = {prob_one_departure_modified:.4f}")

                    # Calculate the new mixed operation capacity using the modified probability for 1 departure
                    sum_prob_n_times_n_modified = 0

                    # Loop through the dictionary to calculate the sum of (prob of n departures * n)
                    for num_departures, total_probability in departure_probabilities_modified.items():
                        sum_prob_n_times_n_modified += total_probability * num_departures

                    Eij_value_modified = np.sum(Tij_matrix_transposed_modified * Pij_matrix)

                    # Calculate the new capacity (Mixed Operation) with the modified matrix
                    capacity_mixed_operation_modified = int(np.round((3600 * (1 + sum_prob_n_times_n_modified)) / Eij_value_modified))

                    # Display the new calculated mixed operation capacity
                    st.subheader(
                        f"New Capacity (Mixed Operation) = {capacity_mixed_operation_modified:} operations/hour")

            except ValueError:
                st.error("Please ensure all inputs are valid numbers.")


if __name__ == "__main__":
    app()
