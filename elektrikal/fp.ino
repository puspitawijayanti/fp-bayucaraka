#include <AccelStepper.h>
#include <Servo.h>
// Define motor connections and interfaces
#define STEP_X 11
#define DIR_X 12
#define STEP_Y 6
#define DIR_Y 3
#define STEP_Z 7
#define DIR_Z 4
#define ENABLE 8
#define SERVO_PIN 9

#define MAX_SPEED 1000
#define ACCELERATION 500

// Initialize stepper motors
AccelStepper stepperX(AccelStepper::DRIVER, STEP_X, DIR_X);
AccelStepper stepperY(AccelStepper::DRIVER, STEP_Y, DIR_Y);
AccelStepper stepperZ(AccelStepper::DRIVER, STEP_Z, DIR_Z);
// Initialize servo
Servo gripper;

void setup() {
  // Set max speed and acceleration
  stepperX.setMaxSpeed(MAX_SPEED);
  stepperX.setAcceleration(ACCELERATION);

  stepperY.setMaxSpeed(MAX_SPEED);
  stepperY.setAcceleration(ACCELERATION);

  stepperZ.setMaxSpeed(400);
  stepperZ.setAcceleration(300);

  // Enable stepper drivers
  pinMode(ENABLE, OUTPUT);
  digitalWrite(ENABLE, LOW);

  // Attach servo
  gripper.attach(SERVO_PIN);
  gripper.write(0);

  // Initialize serial communication
  Serial.begin(9600);
  Serial.println("System Initialized");
}

void loop() {
  // Check for serial data
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    processCommand(command);
  }

  // Run stepper motors to their target positions
  stepperX.run();
  stepperY.run();
  stepperZ.run();
}

void processCommand(const String &command) {
  if (command.length() < 2) {
    Serial.println("Error: Invalid command");
    return;
  }

  char cmdType = command.charAt(0);
  int value = command.substring(1).toInt();

  switch (cmdType) {
    case 'X':
      stepperX.moveTo(value);
      Serial.println("X target set to " + String(value));
      break;
    case 'Y':
      stepperY.moveTo(value);
      Serial.println("Y target set to " + String(value));
      break;
    case 'Z':
      stepperZ.moveTo(value);
      Serial.println("Z target set to " + String(value));
      break;
    case 'S':
      gripper.write(value);
      break;
    default:
      Serial.println("Error: Unknown command");
  }
}
