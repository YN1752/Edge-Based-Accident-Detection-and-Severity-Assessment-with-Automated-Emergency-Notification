import cv2
from ultralytics import YOLO
import time
import sys
import threading
import subprocess
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTShadowClient

def myShadowUpdateCallback(payload, responseStatus, token):
    print()

def scheduled_task(stop_event, interval_seconds, script_name, severity, vehicles_involved):
    
    SHADOW_CLIENT = "myShadowClient10"
    HOST_NAME = "a2opxnqukmgii6-ats.iot.us-east-1.amazonaws.com"
    ROOT_CA = "AmazonRootCA1.pem"
    PRIVATE_KEY = "private.pem.key"
    CERT_FILE = "certificate.pem.crt"
    SHADOW_HANDLER = "RPi_Thing_1752"

    myShadowClient = AWSIoTMQTTShadowClient( SHADOW_CLIENT)
    myShadowClient.configureEndpoint (HOST_NAME, 8883)
    myShadowClient.configureCredentials (ROOT_CA, PRIVATE_KEY,CERT_FILE)
    myShadowClient.configureConnectDisconnectTimeout(10)
    myShadowClient.configureMQTTOperationTimeout(5)
    myShadowClient.connect()
    myDeviceShadow = myShadowClient.createShadowHandlerWithName(SHADOW_HANDLER, True)

    print(f"[{time.ctime()}] Launching {script_name} in a new process...")
    process = subprocess.Popen(["python", script_name, severity, vehicles_involved], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    report, _ = process.communicate()
    message = '{"report": "' + report.strip() + '"}'
    print(message)
    myDeviceShadow.shadowUpdate( message, myShadowUpdateCallback, 5)

script_to_run = 'generate_report.py'
frequency = 10

# --- Configuration ---
CLASSES_TO_PREDICT = [3,4,6] 
CONSECUTIVE_FRAMES_THRESHOLD = 5
VIDEO_SOURCE = f"sample_videos/{sys.argv[1]}" if len(sys.argv)==2 else 0 # 0 for webcam or 'path/to/video.mp4'

# --- Initialization ---
sev_est_model = YOLO('models/sev_est_full_integer_quant_edgetpu.tflite')
veh_det_model = YOLO('models/veh_det1_full_integer_quant_edgetpu.tflite')

cap = cv2.VideoCapture(VIDEO_SOURCE)

#print(sev_est_model.names)

consecutive_frames_count = 0
last_detected_class = None
fps_start_time = time.time()
frame_count = 0
fps=0
fps_list = []

cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Camera Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

thread_flag = True
stop_event = threading.Event()

# --- Main Loop ---
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame_count += 1

    frame = cv2.resize(frame, (640,640))
    
    # Run inference, filtering for specific classes
    results = sev_est_model.predict(frame, classes=CLASSES_TO_PREDICT, verbose=False)
    
    current_detected = False
    if results[0].boxes:
        # Detect the first object's class ID
        detected_class_id = int(results[0].boxes.cls[0].item())
        current_detected = True

    # Check for consecutive frames
    if current_detected and detected_class_id == last_detected_class:
        consecutive_frames_count += 1
    else:
        consecutive_frames_count = 1
        last_detected_class = detected_class_id if current_detected else None

    # Output if threshold met
    if consecutive_frames_count == CONSECUTIVE_FRAMES_THRESHOLD and current_detected:
        severity = sev_est_model.names[detected_class_id]

        # print(f"ALERT: '{severity}' detected for {consecutive_frames_count} frames!")

        # print(results[0].boxes.conf[0].item())

        x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[0].tolist())

        offset = 5

        h, w, _ = frame.shape
        x1, y1 = max(0, x1-offset), max(0, y1-offset)
        x2, y2 = min(w, x2+offset), min(h, y2+offset)

        crop_frame = frame[y1:y2, x1:x2]

        crop_frame = cv2.resize(crop_frame, (800, 600))

        veh_det_results = veh_det_model.predict(crop_frame, verbose=False, conf=0.59)
        vehicles_list = [veh_det_model.names[int(box.cls[0])] for box in veh_det_results[0].boxes]
        
        # print(vehicles_list)
        # print(crop_frame.shape)
        
        vehicles_involved = ", ".join(vehicles_list)

        # print(vehicles_involved)

        # cv2.imshow("Last Frame", veh_det_results[0].plot())

        if thread_flag==True:
            scheduler_thread = threading.Thread(target=scheduled_task, args=(stop_event, frequency, script_to_run, severity, vehicles_involved))
            scheduler_thread.start()
            thread_flag=False

    # --- Display FPS and results ---
    if frame_count >= 10:
        fps = frame_count / (time.time() - fps_start_time)
        fps_list.append(fps)
        frame_count = 0
        fps_start_time = time.time()
    
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera Feed", results[0].plot()) # Visualize results
    if cv2.waitKey(1) & 0xFF == ord("q"): break

stop_event.set()
cap.release()
cv2.destroyAllWindows()

avg_fps = sum(fps_list[1:])/(len(fps_list)-1)
max_fps = max(fps_list[1:])
min_fps = min(fps_list[1:])

print(f"Avg FPS: {avg_fps:.2f}")
print(f"Max FPS: {max_fps:.2f}")
print(f"Min FPS: {min_fps:.2f}")
