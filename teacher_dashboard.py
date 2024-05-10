import os
from flask import Blueprint, render_template

# Create a Blueprint for the teacher's dashboard
teacher_dashboard_blueprint = Blueprint('teacher_dashboard', __name__)

# Define a function to retrieve CSV files from a directory
def get_csv_files(directory):
    csv_files = []
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            csv_files.append(file)
    return csv_files

# Define routes and views for the teacher's dashboard
@teacher_dashboard_blueprint.route('/teacher_dashboard')
def teacher_dashboard():
    # Directory where CSV files are stored
    csv_directory = r'D:\backup\Face-Link-master\Attendance'

    # Retrieve CSV files from the directory
    csv_files = get_csv_files(csv_directory)
    print(csv_files)

    # Render the template with the CSV files
    return render_template('teacher_dashboard.html', csv_files=csv_files)
