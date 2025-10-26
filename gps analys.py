import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
from collections import defaultdict
import json

# --------------------------
# CONFIGURATION & SETUP
# --------------------------
API_KEY = "T56juh1Z8SYQIEK0IDjJI6n5slsStmEh"
CITY_NAME = "Hyderabad"
BBOX = "78.223,17.215,78.602,17.600"
REFRESH_INTERVAL = 120  # seconds

st.set_page_config(
    page_title=f"{CITY_NAME} Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# SESSION STATE
# --------------------------
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'incident_history' not in st.session_state:
    st.session_state.incident_history = []
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

# --------------------------
# TRAFFIC SERVICE
# --------------------------
class TrafficIntelligenceService:
    def __init__(self, api_key):
        self.api_key = api_key
        self.severity_colors = {
            "Accident": "red",
            "Road Closed": "darkred", 
            "Jam": "orange",
            "Road Works": "purple",
            "Broken Vehicle": "lightred",
            "Dangerous Conditions": "beige",
            "Weather": "lightblue",
            "Other": "gray"
        }

    def get_icon(self, category):
        icons = {
            "Accident": "car-crash",
            "Road Closed": "ban-circle",
            "Jam": "time",
            "Road Works": "wrench",
            "Broken Vehicle": "cog",
            "Dangerous Conditions": "warning-sign",
            "Weather": "cloud",
            "Other": "info-sign"
        }
        return icons.get(category, "info-sign")

    def categorize_incident(self, code):
        mapping = {1:"Accident", 6:"Jam", 7:"Road Closed", 8:"Road Closed",
                   9:"Road Works", 14:"Broken Vehicle", 3:"Dangerous Conditions",
                   2:"Weather",4:"Weather",5:"Weather",10:"Weather",11:"Weather",15:"Weather"}
        return mapping.get(code, "Other")

    def fetch_incidents(self, bbox):
        url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
        params = {
            "key": self.api_key,
            "bbox": bbox,
            "fields": "{incidents{type,geometry{type,coordinates},properties{id,iconCategory,magnitudeOfDelay,events{description,code},from,to,delay,length,startTime,endTime}}}",
            "language": "en-GB",
            "timeValidityFilter": "present"
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            incidents = data.get("incidents", [])
            for inc in incidents:
                props = inc.get("properties", {})
                code = props.get("iconCategory", 0)
                inc['category'] = self.categorize_incident(code)
                inc['severity_color'] = self.severity_colors.get(inc['category'], "gray")
                inc['icon'] = self.get_icon(inc['category'])
            return incidents
        except Exception as e:
            st.error(f"Error fetching incidents: {e}")
            return []

# --------------------------
# MAP CREATION
# --------------------------
def create_map(incidents):
    m = folium.Map(location=[17.3850,78.4867], zoom_start=12, control_scale=True)

    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        name="OpenStreetMap",
        attr="© OpenStreetMap contributors"
    ).add_to(m)

    folium.LayerControl().add_to(m)

    for inc in incidents:
        geom = inc["geometry"]
        desc = inc.get("properties", {}).get("events", [{}])[0].get("description", "Traffic incident")
        popup_html = f"<b>{desc}</b><br>Type: {inc['category']}"

        if geom["type"] == "Point":
            coords = geom["coordinates"][::-1]
            folium.Marker(
                location=coords,
                popup=popup_html,  # must be str
                tooltip=desc,      # must be str
                icon=folium.Icon(color=inc.get("color", "gray"), icon="info-sign")
            ).add_to(m)

        elif geom["type"] == "LineString":
            coords = [(lat, lon) for lon, lat in geom["coordinates"]]
            folium.PolyLine(
                coords,
                color=inc.get("color", "gray"),
                weight=6,
                opacity=0.8,
                popup=popup_html  # must be str
            ).add_to(m)

    return m

# --------------------------
# ANALYTICS
# --------------------------
def analytics(incidents):
    if not incidents:
        st.info("No incident data for analytics")
        return
    data=[]
    for inc in incidents:
        props = inc.get("properties",{})
        data.append({
            "category": inc['category'],
            "description": props.get("events",[{}])[0].get("description","Unknown"),
            "delay": props.get("delay",0),
            "length": props.get("length",0)/1000,
            "magnitude": props.get("magnitudeOfDelay","Unknown")
        })
    df = pd.DataFrame(data)
    
    c1,c2,c3 = st.columns(3)
    c1.metric("Total Incidents", len(incidents))
    c2.metric("Total Delay (min)", f"{df['delay'].sum()/60:.0f}")
    c3.metric("Avg Incident Length (km)", f"{df['length'].mean():.1f}")
    
    c1,c2 = st.columns(2)
    cat_counts = df['category'].value_counts()
    fig1 = px.pie(values=cat_counts.values, names=cat_counts.index, 
                  title="Incidents by Category", color_discrete_sequence=px.colors.qualitative.Set3)
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig1, use_container_width=True)
    
    delay_by_cat = df.groupby('category')['delay'].sum().sort_values(ascending=False)
    fig2 = px.bar(x=delay_by_cat.index, y=delay_by_cat.values/60,
                  title="Total Delay by Category (min)", labels={'x':'Category','y':'Delay (min)'},
                  color=delay_by_cat.index, color_discrete_sequence=px.colors.qualitative.Set3)
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

# --------------------------
# SIDEBAR
# --------------------------
def sidebar():
    with st.sidebar:
        st.header("Traffic Controls")
        st.session_state.auto_refresh = st.checkbox("Enable Auto-Refresh", value=st.session_state.auto_refresh)
        if st.button("Refresh Now"):
            st.experimental_rerun()
        st.markdown("---")
        st.header("Incident Legend")
        legend = {"🔴 Accident":"Collision","🟠 Traffic Jam":"Congestion",
                  "🟣 Road Works":"Construction","⚫ Road Closed":"Closed",
                  "🔵 Weather":"Weather issues","🟡 Other":"Miscellaneous"}
        for k,v in legend.items():
            st.caption(f"{k} - {v}")
        st.markdown("---")
        st.header("Statistics")
        st.caption(f"City: {CITY_NAME}")
        st.caption(f"Coverage Area: {BBOX}")
        st.caption(f"Last Update: {st.session_state.last_update.strftime('%H:%M:%S')}")

# --------------------------
# MAIN
# --------------------------
def main():
    traffic_service = TrafficIntelligenceService(API_KEY)
    st.title(f"{CITY_NAME} Traffic Intelligence Dashboard")
    st.markdown("Real-time traffic monitoring with analytics")
    
    sidebar()
    
    if st.session_state.auto_refresh:
        st_autorefresh(interval=REFRESH_INTERVAL*1000, key="auto_refresh")
    
    incidents = traffic_service.fetch_incidents(BBOX)
    st.session_state.incident_history.append({"timestamp":datetime.now(), "incident_count":len(incidents)})
    cutoff = datetime.now() - timedelta(hours=24)
    st.session_state.incident_history = [h for h in st.session_state.incident_history if h["timestamp"]>cutoff]
    
    tab1,tab2,tab3 = st.tabs(["Live Map","Analytics","Trends"])
    
    with tab1:
        st.subheader("Live Traffic Map")
        fol_map = create_map(incidents)
        st_folium(fol_map, width=1200, height=600)
    
    with tab2:
        st.subheader("Traffic Analytics")
        analytics(incidents)
    
    with tab3:
        st.subheader("Incident Trends")
        if len(st.session_state.incident_history)>1:
            df = pd.DataFrame(st.session_state.incident_history)
            fig = px.line(df, x="timestamp", y="incident_count", title="Incident Trends Last 24h")
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.info("Collecting historical data...")
    
    st.markdown("---")
    st.caption(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("Data provided by TomTom Traffic API")

if __name__=="__main__":
    main()
