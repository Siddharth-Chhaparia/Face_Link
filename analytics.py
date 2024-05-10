import pandas as pd
import matplotlib.pyplot as plt
import sys

def load_csv(file_path):
    """
    Load CSV file using pandas.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        return None

def generate_pie_chart(df, column_name):
    """
    Generate pie chart from DataFrame.
    """
    try:
        # Perform analytics on DataFrame to generate data for pie chart
        # For example, count unique values in a column
        data = df[column_name].value_counts()

        # Plot pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Pie Chart')
        plt.show()  # Display the pie chart
    except Exception as e:
        print(f"Error generating pie chart: {str(e)}")

def main():
    # Check if a file path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        return

    file_path = sys.argv[1]

    # Load CSV file
    df = load_csv(file_path)
    if df is not None:
        # Get column name for analytics
        column_name = input("Enter the name of the column for analytics: ")

        # Generate and display pie chart
        generate_pie_chart(df, column_name)

if __name__ == "__main__":
    main()
