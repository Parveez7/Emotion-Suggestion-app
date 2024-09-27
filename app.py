from flask import Flask, render_template, Response, jsonify
import cv2
from keras.models import model_from_json
import numpy as np
import random

app = Flask(__name__)

# Loading the emotion detection model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Suggestions for different emotions
suggestions = {
    'angry': [
        "Take a few deep breaths and count to ten",
        "Go for a brisk walk or do some physical exercise",
        "Write down what's bothering you and possible solutions",
        "Listen to calming music or nature sounds",
        "Practice progressive muscle relaxation"
    ],
    'disgust': [
        "Focus on something pleasant or beautiful in your environment",
        "If it's about a situation, think of ways to improve or change it",
        "Engage in a favorite hobby to distract yourself",
        "Talk to a friend about what's bothering you",
        "Practice mindfulness to center yourself in the present moment"
    ],
    'fear': [
        "Remind yourself that you are safe in this moment",
        "Practice deep breathing exercises",
        "Challenge your fearful thoughts with rational ones",
        "Visualize a peaceful, safe place",
        "Reach out to a supportive friend or family member"
    ],
    'surprise': [
        "Take a moment to process what just happened",
        "Share the surprising event with someone close to you",
        "Use this as an opportunity to learn something new",
        "If it's positive, savor the feeling of unexpected joy",
        "If it's negative, take some deep breaths and assess the situation calmly"
    ],
    'sad': [
        "Take a walk in nature",
        "Listen to your favorite uplifting music",
        "Call a friend or family member",
        "Practice gratitude by listing three things you're thankful for",
        "Try a new hobby or revisit an old one"
    ],
    'neutral': [
        "Set a small goal for today and accomplish it",
        "Learn something new - watch an educational video or read an article",
        "Do a random act of kindness",
        "Take a few deep breaths and practice mindfulness",
        "Plan a fun activity for the near future"
    ],
    'happy': [
        "Share your happiness with others around you",
        "Do something kind for someone else to spread the joy",
        "Take a moment to appreciate and savor this feeling",
        "Engage in an activity you love to maintain the positive mood",
        "Write down what made you happy to remember it later"
    ]
}


# Global variable to store the last detected emotion
last_emotion = 'neutral'

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def get_suggestion(emotion):
    if emotion in ['sad', 'neutral','happy','surprise','fear','disgust','angry']:
        return random.choice(suggestions[emotion])
    return "You seem to be doing well! Keep it up!"

def gen_frames():
    global last_emotion
    webcam = cv2.VideoCapture(0)
    while True:
        success, frame = webcam.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(frame, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                features = extract_features(face)
                prediction = model.predict(features)
                emotion = labels[prediction.argmax()]

                last_emotion = emotion
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_suggestion/<emotion>')
def suggestion(emotion):
    return get_suggestion(emotion)

@app.route('/get_emotion')
def get_emotion():
    global last_emotion
    return jsonify({'emotion': last_emotion})

if __name__ == '__main__':
    app.run(debug=True)