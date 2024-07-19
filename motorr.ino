#include <Arduino.h>

// Define motor control pins
const int LEFT_MOTOR_FORWARD_PIN = 4;
const int LEFT_MOTOR_BACKWARD_PIN = 5;
const int RIGHT_MOTOR_FORWARD_PIN = 6;
const int RIGHT_MOTOR_BACKWARD_PIN = 7;

void setup() {
  Serial.begin(9600);

  // Initialize motor control pins as outputs
  pinMode(LEFT_MOTOR_FORWARD_PIN, OUTPUT);
  pinMode(LEFT_MOTOR_BACKWARD_PIN, OUTPUT);
  pinMode(RIGHT_MOTOR_FORWARD_PIN, OUTPUT);
  pinMode(RIGHT_MOTOR_BACKWARD_PIN, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    if (command == "Start") {
      // Move forward
      digitalWrite(LEFT_MOTOR_FORWARD_PIN, HIGH);
      digitalWrite(RIGHT_MOTOR_FORWARD_PIN, HIGH);
    } else if (command == "Stop") {
      // Stop
      digitalWrite(LEFT_MOTOR_FORWARD_PIN, LOW);
      digitalWrite(RIGHT_MOTOR_FORWARD_PIN, LOW);
      digitalWrite(LEFT_MOTOR_BACKWARD_PIN, LOW);
      digitalWrite(RIGHT_MOTOR_BACKWARD_PIN, LOW);
    }
  }
  delay(100);
}
