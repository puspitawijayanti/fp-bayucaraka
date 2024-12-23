import cv2
import numpy as np
import serial
import time
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class VisionControlNode(Node):
    def __init__(self):
        super().__init__('with_container')

        self.publisher_xy = self.create_publisher(String, 'xy_coordinates', 10)

        self.publisher_z = self.create_publisher(String, 'z_command', 10)

        model_path = "/home/tata/runs/detect/train19/weights/best.pt"    
        self.model = YOLO(model_path)
        self.get_logger().info(f"Loaded YOLO model from {model_path}")

        try:
            self.arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)  
            time.sleep(2)  
            self.get_logger().info("connected to arduino on /dev/ttyACM0")
        except Exception as e:
            self.get_logger().error(f"error initializing serial communication: {e}")
            exit()

        self.camera = cv2.VideoCapture(2) 
        if not self.camera.isOpened():
            self.get_logger().error("unable to access the camera")
            exit()

        self.STEPS_PER_CM_X = 1400 / 42  # ≈ 26.92 steps/cm
        self.STEPS_PER_CM_Y = 2600 / 48  # ≈ 54.17 steps/cm
        self.WORKSPACE_WIDTH = 42  # cm
        self.WORKSPACE_HEIGHT = 48  # cm
        self.MOVEMENT_THRESHOLD = 10
        self.ALIGNMENT_THRESHOLD = 1
        self.CONFIDENCE_THRESHOLD = 0.5  

        # offset gripper sumbu x as milimetrer (mm) 
        self.gripper_offset_mm = 45  
        self.workspace_width_mm = 520  
        self.workspace_height_mm = 500
        # self.zup = False
        
        self.current_position_x = 0
        self.current_position_y = 0
        self.z_triggered = False
        self.gripper_state = False
        self.xy_aligned = False
        self.stop = False

        # Rentang warna kuning
        self.yellow_ranges = [
            {"lower": np.array([20, 100, 100], dtype=np.uint8), "upper": np.array([30, 255, 255], dtype=np.uint8)},  # pure yellow
            {"lower": np.array([15, 50, 200], dtype=np.uint8), "upper": np.array([30, 100, 255], dtype=np.uint8)},   # light yellow
            {"lower": np.array([20, 150, 150], dtype=np.uint8), "upper": np.array([30, 255, 255], dtype=np.uint8)},  # golden yellow
            {"lower": np.array([25, 50, 50], dtype=np.uint8), "upper": np.array([35, 255, 255], dtype=np.uint8)},    # yellow-green
            {"lower": np.array([15, 100, 100], dtype=np.uint8), "upper": np.array([20, 255, 255], dtype=np.uint8)},  # yellow-orange
        ]

    def calculate_steps(self, coord, frame_size, steppercm, workspace_size):
        normalized_coord = coord / frame_size
        cm_position = normalized_coord*workspace_size
        return int(cm_position* steppercm)
    

    def send_command(self, command):
        try:
            self.arduino.write(f"{command}\n".encode())
            time.sleep(0.05)
        except Exception as e:
            self.get_logger().error(f"Error sending command: {e}")

    # Mendeteksi tempat berdasarkan rentang warna kuning.
    def detect_yellow_target(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for color_range in self.yellow_ranges:
            mask = cv2.inRange(hsv_frame, color_range["lower"], color_range["upper"])
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 500:  
                    x, y, w, h = cv2.boundingRect(contour)
                    return x + w // 2, y + h // 2  
        return None, None

    # deteksi dijalankan pakai yolo
    def process_frame(self, frame):
        results = self.model.predict(source=frame, save=False, conf=self.CONFIDENCE_THRESHOLD)

        for result in results:
            boxes = result.boxes
            if not boxes:
                continue

            # bounding box pertama yang memenuhi threshold
            box = None
            for b in boxes:
                if b.conf[0] >= self.CONFIDENCE_THRESHOLD:
                    box = b
                    break

            if box is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            conf = box.conf[0]  # buat confidence
            class_id = int(box.cls[0])  # kelas objek
            class_name = self.model.names.get(class_id, f"Class {class_id}")

            object_center_x = (x1 + x2) // 2
            object_center_y = (y1 + y2) // 2

            # skala konversi mm ke piksel
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]
            mm_to_pixel_ratio_x = frame_width / self.workspace_width_mm
            mm_to_pixel_ratio_y = frame_height / self.workspace_height_mm

            # hitung offset di sumbu x secara piksel
            gripper_offset_x = int(self.gripper_offset_mm * mm_to_pixel_ratio_x)

            # pusat offset di-adjust dengan gripper sumbu x
            adjusted_center_x = object_center_x + gripper_offset_x
            adjusted_center_y = object_center_y

            # posisi motor dihitung berdasarkan deteksi
            target_x = self.calculate_steps(object_center_x, frame_width, self.STEPS_PER_CM_X, self.WORKSPACE_WIDTH)
            target_y = self.calculate_steps(object_center_y, frame_height, self.STEPS_PER_CM_Y, self.WORKSPACE_HEIGHT)-235
            # menghitung jarak dari tengah frame ke objek
            middle_x = frame_width // 2
            middle_y = frame_height // 2
            distance_x = abs(object_center_x - middle_x)
            distance_y = abs(object_center_y - middle_y)
            distance_cm_x = distance_x / mm_to_pixel_ratio_x
            distance_cm_y = distance_y / mm_to_pixel_ratio_y


            # x-y akan digerakkan kalau belum sejajar
            # sumbu xy dibagi menjadi beberapa part berdasarkan stepper
            if not self.xy_aligned:
                if abs(target_x - self.current_position_x) > self.MOVEMENT_THRESHOLD:
                    if target_x <= 350: self.send_command(f"X{target_x-220}")
                    if target_x >350 and target_x <700:  self.send_command(f"X{target_x-80}")
                    if target_x == 700: self.send_command(f"X{target_x}")
                    if target_x > 700 and target_x < 1050: self.send_command(f"X{target_x+80}")
                    if target_x >= 1050 :self.send_command(f"X{target_x+220}")
                    self.current_position_x = target_x

                if abs(target_y - self.current_position_y) > self.MOVEMENT_THRESHOLD:
                    if target_y <= 650: self.send_command(f"Y{target_y-180}")
                    if target_y >650 and target_y <1300:  self.send_command(f"Y{target_y-80}")
                    if target_y == 1300: self.send_command(f"Y{target_y}")
                    if target_y > 1300 and target_y < 1950: self.send_command(f"Y{target_y+80}")
                    if target_y >= 1950 :self.send_command(f"Y{target_y+180}")
                    self.current_position_y = target_y

                # cek sejajar xy
                if (abs(target_x - self.current_position_x) < self.ALIGNMENT_THRESHOLD and
                        abs(target_y - self.current_position_y) < self.ALIGNMENT_THRESHOLD):
                    self.xy_aligned = True
                    self.get_logger().info(f"X-Y aligned at ({self.current_position_x}, {self.current_position_y}).")

            # z turun dari xy sejajar
            if self.xy_aligned and not self.z_triggered:
                time.sleep(3)  # waktu jeda sebelum gerakan berikutnya
                self.send_command("Z210")  # menggerakkan z untuk turun
                self.get_logger().info("Z axis moved down")
                self.z_triggered = True

            # nutup gripper setelah z
            if self.z_triggered and not self.gripper_state:
                time.sleep(2)
                self.send_command("S120")  
                self.get_logger().info("Gripper closed")
                self.gripper_state = True
                time.sleep(5)

            # z diangkat
            if self.gripper_state and self.z_triggered:
                self.send_command("Z0")  
                self.get_logger().info("Z axis moved up")
                time.sleep(5)
                self.send_command("X0")
                self.send_command("Y0")
                self.current_position_x, self.current_position_y = 0, 0
                time.sleep(4)

                # Mencari tempat tujuan berdasarkan warna kuning
                self.get_logger().info("Mencari tempat tujuan berdasarkan warna kuning")
                while True:
                    ret, frame = self.camera.read()
                    if not ret:
                        self.get_logger().error("Failed to grab frame for yellow target")
                        break

                    yellow_x, yellow_y = self.detect_yellow_target(frame)
                    if yellow_x is not None and yellow_y is not None:
                        self.get_logger().info(f"Yellow target found at ({yellow_x}, {yellow_y})")
                        target_x = self.calculate_steps(yellow_x, frame.shape[1], self.STEPS_PER_CM_X, self.WORKSPACE_WIDTH)
                        target_y = self.calculate_steps(yellow_y, frame.shape[0], self.STEPS_PER_CM_Y, self.WORKSPACE_HEIGHT)

                        # self.send_command(f"Y{target_x}")
                        # self.send_command(f"Y{target_y}")
                        if (target_x) < 700 :self.send_command(f"X{target_x-120}")
                        if (target_x) == 700 : self.send_command(f"X{target_x}")
                        if (target_x) > 700 :self.send_command(f"X{target_x+120}")
                        if (target_y) < 1300 :self.send_command(f"Y{target_y-200}")
                        if (target_y) == 1300 : self.send_command(f"Y{target_y}")
                        if (target_y) > 1300 :self.send_command(f"Y{target_y+200}")

                        time.sleep(4)
                        self.send_command("Z210")  # Turunkan Z untuk menaruh objek
                        time.sleep(3)
                        self.send_command("S0")  # Lepaskan gripper
                        time.sleep(2)
                        self.get_logger().info("Object placed at yellow target")
                        self.send_command("Z0")  # Naikkan Z kembali
                        break
                # Kembali ke home setelah meletakkan objek
                self.stop = True
                self.get_logger().info("Kembali ke posisi home")
                self.send_command("X0")
                self.send_command("Y0")
                self.current_position_x, self.current_position_y = 0, 0
                self.get_logger().info("Proses selesai. Sistem berhenti")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f} X:{self.current_position_x}, Y:{self.current_position_y}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def main_loop(self):
        try:
            while rclpy.ok() and not self.stop:
                ret, frame = self.camera.read()
                if not ret:
                    self.get_logger().error("Failed to grab frame")
                    break

                self.process_frame(frame)

                cv2.imshow("FRAME WITH YOLOO", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            self.get_logger().info("Exiting program")

        finally:
            self.camera.release()
            self.get_logger().info("Camera released")
            self.arduino.close()
            self.get_logger().info("Serial connection to Arduino closed")
            cv2.destroyAllWindows()
            self.get_logger().info("OpenCV windows destroyed")


def main(args=None):
    rclpy.init(args=args)
    node = VisionControlNode()

    try:
        node.main_loop()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
