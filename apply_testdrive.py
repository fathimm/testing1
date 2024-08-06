from flask import Flask, render_template, Response, request, jsonify
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np
import torch.nn as nn
import serial

# Define the serial connection to Arduino
ser = serial.Serial('/dev/ttyUSB0', 9600)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, n_class=6):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, n_class)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class BaseResNet(nn.Module):
    def __init__(self, n_class=6):
        super(BaseResNet, self).__init__()
        self.resnet = ResNet(n_class=n_class)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        return x

# Use BaseResNet as the model
model = BaseResNet(n_class=6)

# Define the label mapping
cabe_label = ['Healthy', 'Leaf Curl', 'Leaf Spot', 'Powdery Mildew', 'White Fly', 'Yellowish']

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = ResNet()
model.load_state_dict(torch.load('/home/pi/testing6/Spicy-fe/resnet.pth'))
model.eval()

# Define transformation for the input image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Initialize camera
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/') # root
def index():
    return render_template('index.html') # render file html

@app.route('/video_feed') # nampilin Gambar
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/send-command', methods=['POST']) # command 
def send_command():
    data = request.json
    command = data['command']
    # Handle the command (e.g., send it to Arduino)
    ser.write(command.encode())
    return jsonify({'status': 'success'})

@app.route('/detect-pest') # detection
def detect_pest():
    success, frame = camera.read()
    if success:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_t = transform(img)
        img_t = img_t.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = model(img_t)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            probability = probabilities[0, class_idx].item() * 100

        result = cabe_label[class_idx]
        return jsonify({'result': result, 'probability': f'{probability:.2f}%'})
    else:
        return jsonify({'result': 'No frame detected'})

@app.route('/get-distance') # ultrasonik
def get_distance():
    # Read data from Arduino
    data = ser.readline().decode().strip()
    jarak1, jarak2, jarak3 = data.split(',')
    return jsonify({'jarak1': jarak1, 'jarak2': jarak2, 'jarak3': jarak3}) 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    # Gunain debug=True kalo mau Trace Error ketika mau running flask
    #Example  app.run(host='0.0.0.0', port=5000 debug=True)