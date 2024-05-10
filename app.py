from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
from flask import Flask, request, render_template, url_for
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

from preprocess import preprocesses
import sys
from classifier import training
import facenet
import detect_face
import os
import time
import pickle
from PIL import Image
import tensorflow._api.v2.compat.v1 as tf
from hotspot import get_wifi_ssid, get_third_ipv4_address
from flask import redirect
from flask import redirect, url_for

import cloudinary
from cloudinary.uploader import upload
import mysql.connector

import csv
import json
from flask import jsonify

#### Defining Flask App
app = Flask(__name__)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### Initializing VideoCapture object to access WebCam
# face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')
# cap = cv2.VideoCapture(0)



#### If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')


#### get a number of total registered users
def totalreg():
    return len(os.listdir('train_img'))

#### preprocessing images
def data_preprocess():
    input_datadir = './train_img'
    output_datadir = './allign_img'

    obj=preprocesses(input_datadir,output_datadir)
    nrof_images_total,nrof_successfully_aligned=obj.collect_data()

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)




#### train_main 
def train_main():
    datadir = './allign_img'
    modeldir = './model/20180402-114759.pb'
    classifier_filename = './class/classifier.pkl'
    print ("Training Start")
    obj=training(datadir,modeldir,classifier_filename)
    get_file=obj.main_train()
    print('Saved classifier model to file "%s"' % get_file)
    sys.exit("All Done")


#### face recognition
def face_recog():
    video=0
    modeldir = './model/20180402-114759.pb'
    classifier_filename = './class/classifier.pkl'
    npy = './npy' 
    train_img="./train_img"
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
            minsize = 30  # minimum size of face
            threshold = [0.7,0.8,0.8]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            batch_size =100 #1000
            image_size = 182
            input_image_size = 160
            HumanNames = os.listdir(train_img)
            HumanNames.sort()
            print('Loading Model')
            facenet.load_model(modeldir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile,encoding='latin1')
            
            video_capture = cv2.VideoCapture(video)
            print('Start Recognition')
            while True:
                ret, frame = video_capture.read()
                #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
                timer =time.time()
                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                faceNum = bounding_boxes.shape[0]
                if faceNum > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]
                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    for i in range(faceNum):
                        emb_array = np.zeros((1, embedding_size))
                        xmin = int(det[i][0])
                        ymin = int(det[i][1])
                        xmax = int(det[i][2])
                        ymax = int(det[i][3])
                        try:
                            # inner exception
                            if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                                print('Face is very close!')
                                continue
                            cropped.append(frame[ymin:ymax, xmin:xmax,:])
                            cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                                    interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            if best_class_probabilities>0.87:
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)    #boxing face
                                for H_i in HumanNames:
                                    if HumanNames[best_class_indices[0]] == H_i:
                                        result_names = HumanNames[best_class_indices[0]]
                                        print("Predictions : [ name: {} , accuracy: {:.3f} ]".format(HumanNames[best_class_indices[0]],best_class_probabilities[0]))
                                        cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                                        cv2.putText(frame, result_names, (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 0), thickness=1, lineType=1)
                                        
                            else :
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                                cv2.rectangle(frame, (xmin, ymin-20), (xmax, ymin-2), (0, 255,255), -1)
                                cv2.putText(frame, "?", (xmin,ymin-5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, 0, 0), thickness=1, lineType=1)
                        except:   
                            
                            print("error")
                        
                endtimer = time.time()
                fps = 1/(endtimer-timer)
                cv2.rectangle(frame,(15,30),(135,60),(0,255,255),-1)
                cv2.putText(frame, "fps: {:.2f}".format(fps), (20, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.imshow('Face Recognition', frame)
                key= cv2.waitKey(1)
                if key== 113: # "q"
                    break
            video_capture.release()
            cv2.destroyAllWindows()
    return HumanNames[best_class_indices[0]]        

#### extract the face from an image
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_recog.detectMultiScale(gray, 1.3, 5)
    return face_points





#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, dsize=(50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


#### Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


#### Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')


################## ROUTING FUNCTIONS #########################

#### Our main page
@app.route('/')
def hello_world():
    # Redirect to the admin_selection page
    return redirect(url_for('admin_selection'))

@app.route('/admin_selection')
def admin_selection():
    # Render the admin_selection.html page
    return render_template("admin_selection.html")


database = {'siddharth': '246810', 'ketan': '969289', 'sagar': '997576', 'anish': '997576'}

# Registration page route
@app.route('/registration')
def registration():
    return render_template('registration.html')

# Registration form submission route
@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    # Add the registration data to your database or perform any other necessary action
    # For demonstration purposes, let's assume you're storing the registration data in a dictionary
    database[username] = password
    # Optionally, you can redirect the user to another page after registration
    return render_template('login.html', username=username)


@app.route('/teacher_reg', methods=['GET', 'POST'])
def teacher_reg():
    if request.method == 'POST':
        try:
            # Handle form submission
            username_teacher = request.form['username']
            password_teacher = request.form['password']
            
            # Store the registration data in a dictionary
            teacher_data = {'username': username_teacher, 'password': password_teacher}
            
            # Save the data to a file
            with open('teachers.json', 'a') as file:
                json.dump(teacher_data, file)
                file.write('\n')
            
            # Optionally, you can redirect the user to another page after registration
            return render_template('teacher_login.html', message='Registration successful!')
        except Exception as e:
            # Print or log the error
            print(f"An error occurred: {str(e)}")
            # Optionally, you can render an error page or provide feedback to the user
            return render_template('error.html', error_message='An error occurred during registration.')
    else:
        # Render the registration form
        return render_template("teacher_reg.html")


# Function to load teachers' credentials from JSON file
def load_teachers():
    with open('teachers.json', 'r') as file:
        return json.load(file)

# Flask route for teacher login
@app.route('/teacher_login', methods=['POST', 'GET'])
def teacher_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if either username or password is empty
        if not username or not password:
            return render_template('teacher_login.html', error='Username and password are required!')

        # Load teachers' credentials from the JSON file
        teachers = load_teachers()

        # Check if the provided credentials match any registered teacher
        for teacher_data in teachers:
            if teacher_data['username'] == username and teacher_data['password'] == password:
                # Redirect to the teacher's dashboard upon successful login
                return redirect('/teacher_dashboard?username=' + username)
        
        # If no matching credentials found, show an error message
        return render_template('teacher_login.html', error='Invalid username or password!')
    
    return render_template('teacher_login.html')


from flask import Flask, render_template, send_from_directory, abort
import os
import glob


csv_directory = r'D:\backup\Face-Link-master\Attendance'

@app.route('/teacher_dashboard')
def list_csv_files():
    try:
        csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))  # Use glob for CSV files
        csv_files = [os.path.basename(file) for file in csv_files]  # Extract filenames only
        print(csv)
        return render_template('teacher_dashboard.html', csv_files=csv_files)
    except Exception as e:
        error_message = f"An error occurred while listing CSV files: {str(e)}"
        # Log the error message or handle it appropriately
        print(error_message)
        # Return an error page or message to the user
        abort(500)

@app.route('/download_csv/<path:filename>')
def download_csv(filename):
    try:
        return send_from_directory(csv_directory, filename, as_attachment=True)
    except Exception as e:
        error_message = f"An error occurred while downloading CSV file '{filename}': {str(e)}"
        # Log the error message or handle it appropriately
        print(error_message)
        # Return an error page or message to the user
        abort(500)



@app.route('/login')
def student_login():
    # Render the student_login.html page
    return render_template("login.html")


@app.route('/form_login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        # Get SSID and IPv4 address
        ssid = get_wifi_ssid()
        ipv4_address = get_third_ipv4_address()
        
        # Check if SSID and IPv4 address are obtained successfully
        if ssid and ipv4_address:
            username = request.form.get('username')
            password = request.form.get('password')

            # Check if SSID, IPv4 address, username, and password match
            if (ssid == "DIR-615-CA0E" and ipv4_address == "192.168.0.128" and 
                username in database and database[username] == password):
                # Redirect to the home page after successful login
                return redirect('/home')
            else:
                # Render the login page with an error message
                return render_template('login.html', info='Invalid credentials!')

        else:
            # If SSID or IPv4 address not obtained, display an error message
            return render_template('login.html', info='Unable to fetch SSID or IPv4 address!')

    elif request.method == 'GET':
        # Handle GET request, typically used to display the login form
        return render_template('login.html')

    # Handle other request methods
    return "Method Not Allowed", 405

@app.route('/home', methods=['GET', 'POST'])
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


#### This function will run when we click on Take Attendance Button
@app.route('/takeattendence', methods=['GET'])
def start():

    # if 'face_recognition_model.pkl' not in os.listdir('static'):
    #     return render_template('home.html', totalreg=totalreg(), datetoday2=datetoday2,
    #                            mess='There is no trained model in the static folder. Please add a new face to continue.')

    # cap = cv2.VideoCapture(0)
    # ret = True
    # while ret:
        # ret, frame = cap.read()
        # if extract_faces(frame) != ():
        #     (x, y, w, h) = extract_faces(frame)[0]
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
        #     face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
        #     identified_person = identify_face(face.reshape(1, -1))[0]
        #     add_attendance(identified_person)
        #     cv2.putText(frame, f'{identified_person}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
        #                 cv2.LINE_AA)
        # cv2.imshow('Attendance', frame)
    face_get = face_recog()
    add_attendance(face_get)
    
        # if cv2.waitKey(1) == 27:
        #     break
    # cap.release()
    # cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)


#### This function will run when we add a new user
@app.route('/add', methods=['POST'])
# Main function to add a new user

def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'train_img/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    
    # Capture images
    cap = cv2.VideoCapture(0)
    i = 0
    while i <= 20:
        _, frame = cap.read()
        cv2.waitKey(1000)
        name = newusername + '_' + str(i) + '.jpg'
        cv2.imwrite(userimagefolder + '/' + name, frame)
        cv2.imshow("Adding new User ", frame)
        if cv2.waitKey(10) == 27:
            break
        i += 1
    cap.release()
    cv2.destroyAllWindows()
    
    print('Training Model')
    data_preprocess()
    train_main()
    names, rolls, times, l = extract_attendance()
   
    # Upload images to Cloudinary
    folder_url = upload_folder_to_cloudinary(userimagefolder, 'FACE_LINK')

    #Adding to the database
    add_to_database(newusername, newuserid, folder_url)
    
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                           datetoday2=datetoday2)
    print("added successfully")

cloudinary.config(
  cloud_name = "dksq0v2n4",
  api_key = "956511569853473",
  api_secret = "tg3aTC2GRhzUOC68K8GNxylEZKU"
)

def upload_folder_to_cloudinary(parent_folder_path, cloudinary_folder):
    folder_url = None
    
    for folder_name in os.listdir(parent_folder_path):
        folder_path = os.path.join(parent_folder_path, folder_name)
        if os.path.isdir(folder_path):
            # Check if the folder already exists in Cloudinary
            existing_folder = cloudinary.Search().expression(f'folder={cloudinary_folder+"/"+folder_name}').execute()
            if not existing_folder.get("resources"):
                # Folder doesn't exist, upload images and retrieve folder URL
                for filename in os.listdir(folder_path):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(folder_path, filename)
                        # Upload image to Cloudinary
                        cloudinary.uploader.upload(image_path, folder=cloudinary_folder+"/"+folder_name)
                # Retrieve folder URL from the upload response of the first image
                upload_response = cloudinary.uploader.upload(os.path.join(folder_path, filename), folder=cloudinary_folder+"/"+folder_name)
                folder_url = upload_response.get("secure_url", None)
            else:
                folder_url = existing_folder["resources"][0]["url"]
    
    return folder_url

# Specify the parent folder path and Cloudinary folder here
parent_folder_path = r'D:\backup\Face-Link-master\allign_img'
folder_url = upload_folder_to_cloudinary(parent_folder_path, 'FACE_LINK')

# Print only the URL up to the last "/"
if folder_url:
    print("Folder URL:", folder_url.rsplit('/', 1)[0])
else:
    print("Folder URL not found.")

# Function to add an entry to the database
def add_to_database(newuserid, newusername, folder_url):
    try:
        # Connect to MySQL database
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="root",
            database="facelink"
        )
        
        cursor = db.cursor()

        # Define SQL query
        sql = "INSERT INTO student (ID, Name, URL) VALUES (%s, %s, %s)"
        val = (newuserid, newusername, folder_url)  # Reordered for consistency with SQL query

        # Execute SQL query
        cursor.execute(sql, val)

        # Commit changes and close connection
        db.commit()
        print("Record inserted successfully into MySQL database")
    except mysql.connector.Error as error:
        print("Failed to insert record into MySQL database:", error)
    finally:
        if db.is_connected():
            cursor.close()
            db.close()
            print("MySQL connection is closed")

#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)
