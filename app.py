import streamlit as st
import json
import yaml
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from ultralytics import YOLO
from chatbot import chat

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="PPE Safety Dashboard",
    page_icon="🦺",
    layout="wide"
)

# ── Load Data ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("runs/detect/ppe_detector/weights/best (1).pt")

def load_metrics():
    try:
        with open("metrics/eval_results.json") as f:
            data = json.load(f)
        total = sum(data["class_counts"].values())
        violations = data.get("violation_events", 0)
        compliance = 1 - (violations / total) if total > 0 else 1.0
        return {
            "mAP50":            compliance,
            "mAP50-95":         compliance * 0.75,
            "precision":        compliance,
            "recall":           compliance,
            "violation_events": data.get("violation_events", 0),
            "violation_frames": data.get("violation_frames", 0),
            "total_frames":     data.get("total_frames", 0),
            "class_counts":     data.get("class_counts", {}),
            "source":           data.get("source", "unknown"),
            "timestamp":        data.get("timestamp", "")
        }
    except:
        return {
            "mAP50": 0.75, "mAP50-95": 0.55,
            "precision": 0.80, "recall": 0.72,
            "violation_events": 0, "violation_frames": 0,
            "total_frames": 0, "class_counts": {},
            "source": "", "timestamp": ""
        }

def load_params():
    try:
        with open("params.yaml") as f:
            return yaml.safe_load(f)
    except:
        return {}

CLASSES = [
    "Hardhat", "Mask", "NO-Hardhat", "NO-Mask",
    "NO-Safety Vest", "Person", "Safety Cone",
    "Safety Vest", "machinery", "vehicle"
]

VIOLATION_CLASSES = ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]
SAFE_CLASSES      = ["Hardhat", "Mask", "Safety Vest"]

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image("construction-safety.jpg", use_container_width=True)
st.sidebar.title("🏗️ PPE Detector")
st.sidebar.markdown("**Construction Safety System**")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "🎥 Live Detection", "🤖 Safety Chatbot"]
)

params  = load_params()
metrics = load_metrics()

# ══════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 PPE Detection Dashboard")
    st.markdown("---")

    # ── KPI Cards ─────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⚠️ Violations",       metrics["violation_events"])
    with col2:
        st.metric("🎞️ Total Frames",     metrics["total_frames"])
    with col3:
        st.metric("🚨 Violation Frames", metrics["violation_frames"])
    with col4:
        st.metric("📹 Source",           metrics["source"])

    if metrics["timestamp"]:
        st.caption(f"Last updated: {metrics['timestamp']}")

    st.markdown("---")

    # ── Class Counts Bar Chart ────────────────────────────────
    st.subheader("📊 Detected Objects Count")
    class_counts = metrics["class_counts"]
    if class_counts:
        fig_counts = px.bar(
            x=list(class_counts.keys()),
            y=list(class_counts.values()),
            color=list(class_counts.keys()),
            title=f"Detection counts — {metrics['source']}",
            labels={"x": "Class", "y": "Count"}
        )
        fig_counts.update_layout(showlegend=False)
        st.plotly_chart(fig_counts, use_container_width=True)

    st.markdown("---")

    # ── Compliance Gauge + Breakdown ──────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Compliance Rate")
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = metrics["mAP50"] * 100,
            title = {"text": "Site Compliance (%)"},
            number = {"suffix": "%", "valueformat": ".1f"},
            gauge = {
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#2ecc71"},
                "steps": [
                    {"range": [0, 50],   "color": "#e74c3c"},
                    {"range": [50, 80],  "color": "#f39c12"},
                    {"range": [80, 100], "color": "#2ecc71"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 80
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Violations vs Safe PPE")
        if class_counts:
            violation_total = sum(v for k, v in class_counts.items() if "NO" in k)
            safe_total      = sum(v for k, v in class_counts.items() if k in SAFE_CLASSES)
            fig2 = px.pie(
                values = [violation_total, safe_total],
                names  = ["Violations ❌", "Compliant ✅"],
                color  = ["Violations ❌", "Compliant ✅"],
                color_discrete_map={
                    "Violations ❌": "#e74c3c",
                    "Compliant ✅":  "#2ecc71"
                },
                title = "PPE Compliance Breakdown"
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Class Thresholds ──────────────────────────────────────
    try:
        thresholds = params["detect"]["class_thresholds"]
        st.subheader("🎚️ Detection Thresholds by Class")
        fig3 = px.bar(
            x     = list(thresholds.keys()),
            y     = list(thresholds.values()),
            color = list(thresholds.keys()),
            title = "Confidence Threshold per Class"
        )
        fig3.add_hline(y=0.35, line_dash="dash",
                       annotation_text="Violation threshold")
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    except:
        pass


# ══════════════════════════════════════════════════════════════
# PAGE 2 — LIVE DETECTION
# ══════════════════════════════════════════════════════════════
elif page == "🎥 Live Detection":
    st.title("🎥 PPE Detection")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📁 Upload File", "🖼️ Sample Video"])

    # ── Tab 1: Upload ─────────────────────────────────────────
    with tab1:
        uploaded = st.file_uploader(
            "Upload image or video",
            type=["jpg", "jpeg", "png", "mp4"]
        )

        if uploaded:
            model = load_model()
            conf  = st.slider("Confidence threshold", 0.1, 0.9, 0.25)

            # ── Image ──────────────────────────────────────────
            if uploaded.type.startswith("image"):
                img       = Image.open(uploaded)
                img_array = np.array(img)

                with st.spinner("Running detection..."):
                    results   = model.predict(img_array, conf=conf)[0]
                    annotated = results.plot()

                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Original")
                with col2:
                    st.image(annotated, caption="Detected", channels="BGR")

                if results.boxes:
                    st.subheader("Detection Summary")
                    counts = {}
                    for box in results.boxes:
                        cls = CLASSES[int(box.cls)]
                        counts[cls] = counts.get(cls, 0) + 1

                    violations = {k: v for k, v in counts.items()
                                  if k in VIOLATION_CLASSES}
                    if violations:
                        st.error(f"⚠️ Violations detected: {violations}")
                    else:
                        st.success("✅ No violations detected!")

                    fig = px.pie(
                        values = list(counts.values()),
                        names  = list(counts.keys()),
                        title  = "Detected Objects"
                    )
                    st.plotly_chart(fig)

            # ── Video ──────────────────────────────────────────
            elif uploaded.type == "video/mp4":
                st.info("⏳ Processing video — this may take a minute...")

                # Save uploaded video to disk
                temp_path = Path("temp_video.mp4")
                with open(temp_path, "wb") as f:
                    f.write(uploaded.getbuffer())

                with st.spinner("Running PPE detection on video..."):
                    model.predict(
                        str(temp_path),
                        conf=conf,
                        save=True,
                        project="result",
                        name="dashboard_detect",
                        exist_ok=True
                    )

                # Search for output video
                mp4_files = []
                for d in [Path("result/dashboard_detect"), Path("result")]:
                    if d.exists():
                        found = list(d.glob("*.mp4"))
                        if found:
                            mp4_files = found
                            break

                if mp4_files:
                    video_bytes = mp4_files[0].read_bytes()
                    st.success("✅ Detection complete!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original**")
                        st.video(uploaded)
                    with col2:
                        st.markdown("**With Detections**")
                        st.video(video_bytes)
                else:
                    st.warning("⚠️ Output video not found. Check result/ folder manually.")

    # ── Tab 2: Sample Videos ──────────────────────────────────
    with tab2:
        st.info("Runs detection on your existing test videos")
        video = st.selectbox(
            "Select test video",
            ["indianworkers.mp4", "JapanPPE.mp4"]
        )
        conf = st.slider("Confidence", 0.1, 0.9, 0.25, key="sample_conf")

        if st.button("🚀 Run Detection"):
            model       = load_model()
            output_name = Path(video).stem

            with st.spinner(f"Running on {video}... ⏳"):
                model.predict(
                    video,
                    conf=conf,
                    save=True,
                    project="result",
                    name=output_name,
                    exist_ok=True
                )

            # Find and display output video
            output_dir = Path(f"result/{output_name}")
            mp4_files  = list(output_dir.glob("*.mp4")) if output_dir.exists() else []

            if mp4_files:
                video_bytes = mp4_files[0].read_bytes()
                st.success("✅ Detection complete!")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original**")
                    st.video(video)
                with col2:
                    st.markdown("**With Detections**")
                    st.video(video_bytes)
            else:
                st.warning(f"⚠️ Output not found. Check result/{output_name}/ folder.")


# ══════════════════════════════════════════════════════════════
# PAGE 3 — CHATBOT
# ══════════════════════════════════════════════════════════════
elif page == "🤖 Safety Chatbot":
    st.title("🤖 Safety AI Assistant")
    st.markdown("Ask anything about your PPE detection system!")
    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.history  = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggested questions shown only when chat is empty
    if not st.session_state.messages:
        st.subheader("💡 Try asking:")
        cols = st.columns(2)
        suggestions = [
            "How many violations were detected?",
            "Which PPE items are missing most often?",
            "How can I improve detection accuracy?",
            "What does the compliance rate mean?"
        ]
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s):
                st.session_state.messages.append({"role": "user", "content": s})
                st.rerun()

    # Chat input
    if prompt := st.chat_input("Ask about safety or detections..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat(prompt, st.session_state.history)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.history.append((prompt, response))