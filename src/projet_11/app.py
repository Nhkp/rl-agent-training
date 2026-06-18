import pandas as pd
import streamlit as st
from stable_baselines3 import PPO

from projet_11.evaluate import play_episode
from projet_11.settings import MODEL_PATH, RESULTS_PATH, VIDEO_PATH
from projet_11.video import record_episode_video


st.set_page_config(page_title="Eagle-1 Mission Dashboard", layout="wide")
st.title("Eagle-1 LunarLander Mission")

model_available = MODEL_PATH.exists()
results_available = RESULTS_PATH.exists()

left, right = st.columns(2)
left.metric("Model available", "yes" if model_available else "no")
right.metric("Results available", "yes" if results_available else "no")

if results_available:
    results = pd.read_csv(RESULTS_PATH)
    st.subheader("Experiment Results")
    st.dataframe(results, width="stretch")

    if {"experiment", "mean_reward"}.issubset(results.columns):
        chart_data = results.set_index("experiment")[["mean_reward"]]
        st.line_chart(chart_data)

    if {"experiment", "std_reward"}.issubset(results.columns):
        st.bar_chart(results.set_index("experiment")[["std_reward"]])
else:
    st.info("Run notebooks/mission.ipynb to generate artifacts/mission/results.csv.")

st.subheader("Agent Replay")

if not model_available:
    st.warning("Train and save the best model before replaying a mission.")
else:
    model = PPO.load(MODEL_PATH)

    if st.button("Run one evaluation episode"):
        episode = play_episode(model)
        st.metric("Episode reward", f"{episode['total_reward']:.2f}")
        st.metric("Episode steps", episode["steps"])
        st.write("Actions:", episode["actions"][:100])

    if st.button("Generate or refresh MP4 replay"):
        path = record_episode_video()
        st.success(f"Video written to {path}")

    if VIDEO_PATH.exists():
        st.video(str(VIDEO_PATH))
    else:
        st.info("Generate a replay video to display it here.")
