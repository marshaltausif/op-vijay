import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import tempfile
import matplotlib.pyplot as plt

# ==================== Streamlit page setup ====================
st.set_page_config(layout="wide", page_title="Crowd Safety Dashboard")
st.title("🔥 Crowd Safety Intelligence Dashboard")

# ==================== Video upload ====================
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
else:
    st.warning("Please upload a video file to start.")
    st.stop()

# ==================== Load YOLO model ====================
model = YOLO("models/yolov8n.pt")
if torch.cuda.is_available():
    model.to('cuda')
else:
    model.to('cpu')

# ==================== Session state ====================
if 'running' not in st.session_state:
    st.session_state.running = False

start_button = st.button("Start Processing")
stop_button = st.button("Stop Processing")

if start_button:
    st.session_state.running = True
if stop_button:
    st.session_state.running = False

# ==================== Video setup ====================
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    st.error("Cannot read video file!")
    st.stop()

# ==================== Initialize variables ====================
heatmap_accum = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
people_history = []
frame_count = 0
MAX_PEOPLE_THRESHOLD = 100
FRAME_SKIP = 3
scale = 0.5

# ==================== Metrics horizontal placeholders ====================
col_metrics = st.columns(3)
metric1_placeholder = col_metrics[0].empty()
metric2_placeholder = col_metrics[1].empty()
metric3_placeholder = col_metrics[2].empty()

alert_placeholder = st.empty()

# Panels layout (top row, remove empty middle gap)
col1, col2, col3 = st.columns([1.4,1.4,1.4])
frame_placeholder = col1.empty()
heatmap_placeholder = col2.empty()
zone_placeholder = col3.empty()

# Big full-width graph at bottom
st.markdown("---")
big_graph_placeholder = st.empty()

# ==================== Zone grid setup ====================
NUM_ROWS = 2
NUM_COLS = 3
ZONE_COLOR_LOW = (0,255,0)
ZONE_COLOR_MED = (0,165,255)
ZONE_COLOR_HIGH = (255,0,0)

zone_h = frame.shape[0] // NUM_ROWS
zone_w = frame.shape[1] // NUM_COLS
zones = []
for r in range(NUM_ROWS):
    for c in range(NUM_COLS):
        x1 = c*zone_w
        y1 = r*zone_h
        x2 = x1+zone_w
        y2 = y1+zone_h
        zones.append((x1,y1,x2,y2))

# ==================== Main processing loop ====================
plt.style.use('dark_background')
while st.session_state.running:
    ret, frame = cap.read()
    if not ret:
        st.info("Video processing completed.")
        st.session_state.running = False
        break

    if frame_count % FRAME_SKIP != 0:
        frame_count += 1
        continue

    frame_small = cv2.resize(frame, (0,0), fx=scale, fy=scale)
    results = model(frame_small, verbose=False)[0]

    # ===== People count =====
    people_count = sum(1 for cls in results.boxes.cls if int(cls)==0)
    people_history.append(people_count)

    # ===== Bounding boxes =====
    display_frame = frame.copy()
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, np.array(box)/scale)
        label = model.names[int(cls)]
        color = (0,255,0) if label=="person" else (0,0,255)
        cv2.rectangle(display_frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(display_frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ===== Heatmap =====
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls)==0:
            x1, y1, x2, y2 = map(int, np.array(box)/scale)
            heatmap[y1:y2, x1:x2] += 1
    heatmap_accum = cv2.addWeighted(heatmap_accum,0.9,heatmap,0.1,0)
    heatmap_normalized = cv2.normalize(heatmap_accum,None,0,255,cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    overlay_frame = cv2.addWeighted(display_frame,0.7,heatmap_color,0.3,0)

    # ===== Density & Risk =====
    density_score = min(1.0, people_count/MAX_PEOPLE_THRESHOLD)
    if people_count>60 or density_score>0.6:
        risk_text="HIGH"
        risk_color=(255,0,0)
    elif people_count>30 or density_score>0.3:
        risk_text="MEDIUM"
        risk_color=(0,165,255)
    else:
        risk_text="LOW"
        risk_color=(0,255,0)

    # ===== Update metrics properly in place =====
    metric1_placeholder.metric("People Count", people_count)
    metric2_placeholder.metric("Density Score", f"{density_score:.2f}")
    metric3_placeholder.metric("Risk Level", risk_text)

    alert_placeholder.markdown(
        f"### **RISK ALERT:** <span style='color:rgb{risk_color}'>{risk_text}</span>",
        unsafe_allow_html=True
    )

    # ===== Update panels =====
    frame_placeholder.image(cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB),
                            caption="YOLO Detection + Density", use_column_width=True)
    heatmap_placeholder.image(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB),
                              caption="Crowd Density Heatmap", use_column_width=True)

    # ===== Zone-wise panel =====
    zone_img = np.zeros_like(frame)
    for idx, (x1,y1,x2,y2) in enumerate(zones):
        count_in_zone = 0
        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            if int(cls)!=0: 
                continue
            bx1, by1, bx2, by2 = map(int, np.array(box)/scale)
            cx = (bx1+bx2)//2
            cy = (by1+by2)//2
            if x1 <= cx < x2 and y1 <= cy < y2:
                count_in_zone += 1

        # Color per count
        if count_in_zone > 10:
            color = ZONE_COLOR_HIGH
        elif count_in_zone > 5:
            color = ZONE_COLOR_MED
        else:
            color = ZONE_COLOR_LOW

        cv2.rectangle(zone_img, (x1,y1), (x2,y2), color, -1)
        cv2.putText(zone_img, f"Zone {idx+1}: {count_in_zone}", (x1+5, y1+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0),2)

    zone_overlay = cv2.addWeighted(frame,0.6,zone_img,0.4,0)
    zone_placeholder.image(cv2.cvtColor(zone_overlay, cv2.COLOR_BGR2RGB),
                           caption="Zone-wise Crowd Density", use_column_width=True)

    # ===== Full-width graph (transparent background) =====
    fig2, ax2 = plt.subplots(figsize=(12,4), facecolor="none")
    ax2.plot(people_history, color='lime', linewidth=2)
    ax2.set_title("People Count Over Time", color="white")
    ax2.set_xlabel("Frame", color="white")
    ax2.set_ylabel("People", color="white")
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("white")
    fig2.patch.set_alpha(0.0)  # fully transparent
    big_graph_placeholder.pyplot(fig2, clear_figure=True)

    frame_count += 1

cap.release()
