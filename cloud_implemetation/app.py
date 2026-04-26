import streamlit as st
import pandas as pd
import cv2
from ultralytics import YOLO
import time
import threading
import subprocess
import tempfile
import os
import atexit
from AWSIoTPythonSDK.exception.AWSIoTExceptions import publishQueueDisabledException
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTShadowClient
import sys

# ---------------- AWS FUNCTION ----------------
def myShadowUpdateCallback(payload, responseStatus, token):
    print()

def scheduled_task(stop_event, interval_seconds, script_name, severity, vehicles_involved):

    SHADOW_CLIENT='myShadowClient10'
    HOST_NAME='a2opxnqukmgii6-ats.iot.us-east-1.amazonaws.com'
    ROOT_CA='AmazonRootCA1.pem'
    PRIVATE_KEY='private.pem.key'
    CERT_FILE='certificate.pem.crt'
    SHADOW_HANDLER='RPi_Thing_1752'

    myShadowClient = AWSIoTMQTTShadowClient(SHADOW_CLIENT)
    myShadowClient.configureEndpoint(HOST_NAME, 8883)
    myShadowClient.configureCredentials(ROOT_CA, PRIVATE_KEY, CERT_FILE)
    myShadowClient.configureConnectDisconnectTimeout(10)
    myShadowClient.configureMQTTOperationTimeout(5)
    myShadowClient.connect()

    myDeviceShadow = myShadowClient.createShadowHandlerWithName(SHADOW_HANDLER, True)

    try:
        process = subprocess.Popen(['python', script_name, severity, vehicles_involved], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        report, _ = process.communicate()
        message = '{"report": "' + report.strip() + '"}'
        myDeviceShadow.shadowUpdate(message, myShadowUpdateCallback, 5)
    except publishQueueDisabledException:
        sys.exit(0)

CLASSES_TO_PREDICT=[3,4,6]
CONSECUTIVE_FRAMES_THRESHOLD=5
script_to_run='generate_report.py'
frequency=10

@st.cache_resource
def load_models():
    return YOLO('models/sev_est.pt'), YOLO('models/veh_det1.pt')

sev_est_model, veh_det_model = load_models()

if 'stop_event' not in st.session_state:
    st.session_state.stop_event = threading.Event()
if 'running' not in st.session_state:
    st.session_state.running = False

atexit.register(lambda: st.session_state.stop_event.set())

st.title('Accident Detection System')

video_file=st.file_uploader('Upload a video for inference', type=['mp4','avi','mov'])

col1,col2=st.columns(2)
start=col1.button('Start Inference', disabled=st.session_state.running)
stop=col2.button('Stop')

if stop:
    st.session_state.stop_event.set()
    st.session_state.running=False

perf_placeholder=st.empty()

def show_perf(fps_list):
    if len(fps_list) <= 1:
        return
    vals=fps_list[1:]
    df=pd.DataFrame([{'Avg. FPS':f'{sum(vals)/len(vals):.2f}','Max FPS':f'{max(vals):.2f}','Min FPS':f'{min(vals):.2f}'}])
    perf_placeholder.subheader('Performance:')
    perf_placeholder.write(df)

def run_inference(cap):
    st.session_state.running=True
    st.session_state.stop_event.clear()

    frame_placeholder=st.empty()
    consecutive_frames_count=0
    last_detected_class=None
    fps_start_time=time.time(); frame_count=0; fps=0; fps_list=[]; thread_flag=True

    while cap.isOpened() and not st.session_state.stop_event.is_set():
        success, frame = cap.read()
        if not success: break
        
        frame_count += 1
        frame = cv2.resize(frame,(640,640))
        
        results = sev_est_model.predict(frame, classes=CLASSES_TO_PREDICT, verbose=False)
        current_detected=False
        
        if results[0].boxes:
            detected_class_id=int(results[0].boxes.cls[0].item())
            current_detected=True
        
        if current_detected and detected_class_id == last_detected_class:
            consecutive_frames_count += 1
        else:
            consecutive_frames_count = 1
            last_detected_class = detected_class_id if current_detected else None
        
        if consecutive_frames_count == CONSECUTIVE_FRAMES_THRESHOLD and current_detected:
            severity = sev_est_model.names[detected_class_id]

            x1,y1,x2,y2 = map(int, results[0].boxes.xyxy[0].tolist())
            offset=5; h,w,_=frame.shape
            x1,y1=max(0,x1-offset),max(0,y1-offset); x2,y2=min(w,x2+offset),min(h,y2+offset)
            
            crop_frame=cv2.resize(frame[y1:y2,x1:x2],(800,600))
            
            veh_det_results=veh_det_model.predict(crop_frame, verbose=False, conf=0.59)
            vehicles_list = [veh_det_model.names[int(box.cls[0])] for box in veh_det_results[0].boxes]

            if len(vehicles_list)<2:
                vehicles_list.append("car")

            vehicles_involved=', '.join(vehicles_list)
            
            if thread_flag:
                threading.Thread(target=scheduled_task,args=(st.session_state.stop_event,frequency,script_to_run,severity,vehicles_involved)).start()
                thread_flag=False

        if frame_count >= 10:
            fps=frame_count/(time.time()-fps_start_time)
            fps_list.append(fps)
            frame_count=0; fps_start_time=time.time()

        annotated_frame=results[0].plot()
        cv2.putText(annotated_frame,f'FPS: {fps:.2f}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        annotated_frame=cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        frame_placeholder.image(annotated_frame, channels='RGB')

    cap.release()
    st.session_state.stop_event.set()
    st.session_state.running=False
    show_perf(fps_list)

if start and video_file is not None:
    tfile=tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    tfile.close()

    run_inference(cv2.VideoCapture(tfile.name))
    os.unlink(tfile.name)