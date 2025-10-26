import streamlit as st
import requests
import pandas as pd
import numpy as np
from math import pow
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="Crowd Sensors Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🌡️ Live Environmental Dashboard</h1>', unsafe_allow_html=True)

# ---------- Sidebar: Location & Settings ----------
st.sidebar.header("🌍 Location Settings")
locations = {
    "🇮🇳 Hyderabad": (17.384, 78.4564),
    "🇮🇳 Mumbai": (19.0760, 72.8777),
    "🇮🇳 Delhi": (28.6139, 77.2090),
    "🇮🇳 Bangalore": (12.9716, 77.5946),
    "🇮🇳 Chennai": (13.0827, 80.2707)
}
selected_city = st.sidebar.selectbox("Choose a city", list(locations.keys()))
latitude, longitude = locations[selected_city]

st.sidebar.header("⚙️ Display Settings")
show_raw_data = st.sidebar.checkbox("Show raw data", value=False)
alert_severity = st.sidebar.slider("Alert Sensitivity", 1, 5, 3)

city_name = selected_city.split(" ")[-1].strip()
st.info(f"📍 **Displaying live environmental data for: {city_name}** (Lat: {latitude}, Lon: {longitude})")

# ---------- Fetch Weather Data ----------
@st.cache_data(ttl=600)
def fetch_weather_data(lat, lon):
    url_weather = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,rain&"
        f"current=temperature_2m,relative_humidity_2m,wind_speed_10m,is_day,rain&"
        f"timezone=auto"
    )
    try:
        response = requests.get(url_weather, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to fetch weather data: {e}")
        return None

data = fetch_weather_data(latitude, longitude)

# ---------- WAQI API Token ----------
WAQI_TOKEN = "dfd4881284ef80636564270d6a5dd10118839e51"  # replace with your token

# ---------- Fetch AQI Data ----------
@st.cache_data(ttl=300)
def fetch_aqi_data(city_name, token):
    url_aqi = f"https://api.waqi.info/feed/{city_name}/?token={token}"
    try:
        response = requests.get(url_aqi, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to fetch air quality data: {e}")
        return None

aqi_data = fetch_aqi_data(city_name, WAQI_TOKEN)

# ---------- Weather Section ----------
if data:
    current = data["current"]
    hourly = data["hourly"]

    temp = current["temperature_2m"]
    humidity = current["relative_humidity_2m"]
    wind_speed = current["wind_speed_10m"]
    is_day = current["is_day"]
    rain = current.get("rain", 0)

    def calculate_heat_index(temp, humidity):
        if temp < 27 or humidity < 40:
            return temp
        HI = (-8.784695 + 1.61139411 * temp + 2.338549 * humidity
              - 0.14611605 * temp * humidity - 0.012308094 * (temp**2)
              - 0.016424828 * (humidity**2) + 0.002211732 * (temp**2) * humidity
              + 0.00072546 * temp * (humidity**2) - 0.000003582 * (temp**2) * (humidity**2))
        return round(HI, 1)

    HI = calculate_heat_index(temp, humidity)

    # ---------- Current Conditions ----------
    st.subheader("📊 Current Conditions")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🌡️ Temperature", f"{temp}°C")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("💧 Humidity", f"{humidity}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🌬️ Wind Speed", f"{wind_speed} m/s")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🔥 Heat Index", f"{HI}°C")
        st.markdown('</div>', unsafe_allow_html=True)

    col5, col6, col7 = st.columns(3)
    with col5:
        st.metric("🌧️ Rain", f"{rain} mm")
    with col6:
        daytime_status = "☀️ Daytime" if is_day else "🌙 Nighttime"
        st.metric("⏰ Time of Day", daytime_status)
    with col7:
        if 18 <= temp <= 26 and 40 <= humidity <= 70:
            comfort = "😊 Comfortable"
        elif HI > 35:
            comfort = "😰 Hot"
        elif temp < 10:
            comfort = "🥶 Cold"
        else:
            comfort = "😐 Moderate"
        st.metric("😊 Comfort Level", comfort)

    # ---------- Forecast Charts ----------
    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "wind_speed": hourly["wind_speed_10m"],
        "rain": hourly.get("rain", [0]*len(hourly["time"]))
    })
    df["heat_index"] = df.apply(lambda x: calculate_heat_index(x["temperature"], x["humidity"]), axis=1)
    now = pd.Timestamp.now().floor('H')
    forecast_df = df[df["time"] >= now].head(12)

    st.markdown("---")
    st.subheader("📈 12-Hour Forecast")
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature & Heat Index', 'Humidity', 'Wind Speed', 'Rainfall'),
        vertical_spacing=0.12
    )

    fig.add_trace(go.Scatter(x=forecast_df["time"], y=forecast_df["temperature"], name="Temperature", line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast_df["time"], y=forecast_df["heat_index"], name="Heat Index", line=dict(color='orange', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=forecast_df["time"], y=forecast_df["humidity"], name="Humidity", line=dict(color='blue'), fill='tozeroy'), row=1, col=2)
    fig.add_trace(go.Scatter(x=forecast_df["time"], y=forecast_df["wind_speed"], name="Wind Speed", line=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Bar(x=forecast_df["time"], y=forecast_df["rain"], name="Rainfall", marker_color='lightblue'), row=2, col=2)

    fig.update_layout(height=600, showlegend=True, title_text="12-Hour Environmental Forecast")
    st.plotly_chart(fig, use_container_width=True)

    # ---------- AQI Section ----------
    st.markdown("---")
    st.subheader("🌫️ Air Quality Index (Live)")

    if aqi_data and aqi_data.get("status") == "ok":
        iaqi = aqi_data["data"]["iaqi"]
        aqi_value = aqi_data["data"]["aqi"]
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("AQI", f"{aqi_value}")
        col2.metric("PM2.5", f"{iaqi.get('pm25', {}).get('v', 'N/A')}")
        col3.metric("PM10", f"{iaqi.get('pm10', {}).get('v', 'N/A')}")
        col4.metric("O₃", f"{iaqi.get('o3', {}).get('v', 'N/A')}")
        col5.metric("NO₂", f"{iaqi.get('no2', {}).get('v', 'N/A')}")
        col6.metric("CO", f"{iaqi.get('co', {}).get('v', 'N/A')}")
    else:
        st.warning("❌ Unable to fetch AQI data")

    # ---------- Live AQI Map ----------
    st.markdown("---")
    st.subheader("🗺️ Live AQI Map")
    m = folium.Map(location=[latitude, longitude], zoom_start=12)
    tile_url = f"https://tiles.aqicn.org/tiles/usepa-aqi/{{z}}/{{x}}/{{y}}.png?token={WAQI_TOKEN}"
    folium.TileLayer(
        tiles=tile_url,
        attr="WAQI",
        name="Air Quality Index",
        overlay=True,
        control=True
    ).add_to(m)
    folium.Marker(location=[latitude, longitude], popup=f"{city_name}", icon=folium.Icon(color="red")).add_to(m)
    st_folium(m, width=700, height=500)

    # ---------- Alerts & Recommendations ----------
    st.markdown("---")
    st.subheader("🚨 Environmental Alerts & Recommendations")
    alerts = []
    recommendations = []

    if temp > 38: alerts.append(("🔥 High Temperature Warning", "Temperature exceeds 38°C - High risk", "high"))
    elif temp > 35: alerts.append(("⚠️ Elevated Temperature", "Temperature above 35°C - Moderate heat risk", "medium"))
    if HI > 42: alerts.append(("🚨 Extreme Heat", f"Heat index {HI}°C - Dangerous", "high"))
    elif HI > 38: alerts.append(("🔥 High Heat", f"Heat index {HI}°C - Uncomfortable", "medium"))
    if humidity > 85: alerts.append(("💧 High Humidity", "Humidity above 85%", "medium"))
    elif humidity < 30: alerts.append(("🏜️ Low Humidity", "Humidity below 30%", "low"))
    if wind_speed < 1: alerts.append(("🌬️ Stagnant Air", "Very low wind", "low"))
    elif wind_speed > 8: alerts.append(("💨 Strong Winds", "High wind speed", "medium"))
    if rain > 5: alerts.append(("🌧️ Heavy Rain", "Significant rainfall", "medium"))
    elif rain > 0: alerts.append(("🌦️ Light Rain", "Light rainfall", "low"))
    if not is_day: recommendations.append("🌙 Nighttime: Ensure adequate lighting")

    if HI > 35: recommendations.append("💧 Stay hydrated and avoid sun exposure")
    if temp < 15: recommendations.append("🧥 Dress warmly for cold conditions")
    if humidity > 80: recommendations.append("💨 Use fans or AC for ventilation")

    for title, message, severity in alerts:
        if severity == "high": st.error(f"**{title}**: {message}")
        elif severity == "medium": st.warning(f"**{title}**: {message}")
        else: st.info(f"**{title}**: {message}")
    if not alerts: st.success("✅ All environmental conditions normal")

    if recommendations:
        st.subheader("💡 Recommendations")
        for rec in recommendations:
            st.write(f"• {rec}")

    # ---------- Raw Forecast Data ----------
    if show_raw_data:
        st.markdown("---")
        st.subheader("📋 Raw Forecast Data")
        with st.expander("View raw forecast data"):
            st.dataframe(forecast_df.style.format({
                "temperature": "{:.1f}",
                "humidity": "{:.1f}",
                "wind_speed": "{:.1f}",
                "heat_index": "{:.1f}",
                "rain": "{:.1f}"
            }))

    # ---------- Footer ----------
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: gray;'>Data provided by Open-Meteo & WAQI APIs | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", 
        unsafe_allow_html=True
    )

else:
    st.error("Unable to load weather data. Please check your internet connection.")
