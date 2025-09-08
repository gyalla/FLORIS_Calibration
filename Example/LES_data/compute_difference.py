import csv

# Function to read numbers from a CSV file
def read_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        # Read the first row and convert to float
        numbers = [float(num) for num in next(reader)]
    return numbers

# Function to write numbers to a CSV file
def write_csv(file_path, data):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

# Main function
def main():
    # File paths
    #file1 = 'MedWS_LowTI_Pulse_A4_St0p3_6D_45.csv'  
    #file2 = 'MedWS_LowTI_Baseline_6D_45.csv'  
    #output_file = 'MedWS_LowTI_Pulse_A4_St0p3_Minus_Baseline_6D_45.csv'  # Output file path

    #file1 = 'MedWS_LowTI_Pulse_A2_St0p3_6D_45.csv'  
    #file2 = 'MedWS_LowTI_Baseline_6D_45.csv'  
    #output_file = 'MedWS_LowTI_Pulse_A2_St0p3_Minus_Baseline_6D_45.csv'  # Output file path

    # file1 = 'MedWS_LowTI_Pulse_A4_St0p3_5D_45.csv'  
    # file2 = 'MedWS_LowTI_Baseline_5D_45.csv'  
    # output_file = 'MedWS_LowTI_Pulse_A4_St0p3_Minus_Baseline_5D_45.csv'  # Output file path

    #file1 = 'MedWS_LowTI_Pulse_A4_St0p3_5D_0.csv'  
    #file2 = 'MedWS_LowTI_Baseline_5D_0.csv'  
    #output_file = 'MedWS_LowTI_Pulse_A4_St0p3_Minus_Baseline_5D_0.csv'  # Output file path

    file1 = 'MedWS_LowTI_Pulse_A4_St0p3_5D_26.csv'  
    file2 = 'MedWS_LowTI_Baseline_5D_26.csv'  
    output_file = 'MedWS_LowTI_Pulse_A4_St0p3_Minus_Baseline_5D_26.csv'  # Output file path

    # Read numbers from the CSV files
    numbers1 = read_csv(file1)
    numbers2 = read_csv(file2)

    # Check if both lists have the same length
    if len(numbers1) != len(numbers2):
        print("Error: The two files must contain the same number of elements.")
        return

    # Subtract the arrays
    result = [num1 - num2 for num1, num2 in zip(numbers1, numbers2)]

    # Write the result to a new CSV file
    write_csv(output_file, result)

    print(f"Subtraction results written to {output_file}")

if __name__ == "__main__":
    main()

