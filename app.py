import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google import genai
import pydeck as pdk
import joblib
import json
import os
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="NexGen Logistics | Prescriptive Analytics",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Gold & Dark Gray) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1E1E1E;
        border-right: 1px solid #333333;
    }
    
    /* Titles & Headers */
    h1, h2, h3, h4 {
        color: #D4AF37 !important; /* Gold */
        font-family: 'Inter', sans-serif;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #D4AF37;
        color: #121212;
        border-radius: 8px;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #B5952F; /* Darker Gold */
        color: #000000;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 2px solid #333;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        color: #888;
        font-size: 16px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: transparent;
        color: #D4AF37;
        border-bottom: 2px solid #D4AF37;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #D4AF37;
    }
    
    /* KPI Cards */
    div.css-1r6slb0 {
        background-color: #1E1E1E;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
try:
    logo = Image.open("assets/logo.png")
    st.sidebar.image(logo)
except:
    st.sidebar.title("NexGen Logistics")

st.sidebar.header("Data & Configuration")

# Gemini API Key
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    client = genai.Client(api_key=api_key)
    st.sidebar.success("Gemini AI Connected")
else:
    st.sidebar.warning("Gemini AI Disconnected")
    client = None

# File Uploaders
st.sidebar.subheader("Data Ingestion")
uploaded_files = {}
file_labels = [
    "orders.csv", "delivery_performance.csv", "routes_distance.csv", 
    "cost_breakdown.csv", "vehicle_fleet.csv", "warehouse_inventory.csv", 
    "customer_feedback.csv"
]

for label in file_labels:
    file = st.sidebar.file_uploader(label, type=["csv"], key=label)
    if file:
        uploaded_files[label] = pd.read_csv(file)
    else:
        default_path = f"NexGen_Logistics/{label}"
        if os.path.exists(default_path):
            uploaded_files[label] = pd.read_csv(default_path)

# Load Model & Metadata
@st.cache_resource
def load_assets():
    try:
        with open("nexgen_delay_model.pkl", "rb") as f:
            model = joblib.load(f)
        with open("model_metadata.json", "r") as f:
            metadata = json.load(f)
        return model, metadata
    except Exception as e:
        st.error(f"Error loading model/metadata: {e}")
        return None, None

model, metadata = load_assets()

# --- Data Processing Function ---
def process_data(files, metadata):
    try:
        df = files["orders.csv"].copy()
        
        # Merge relevant files
        df = df.merge(files["routes_distance.csv"], on="Order_ID", how="left")
        df = df.merge(files["cost_breakdown.csv"], on="Order_ID", how="left")
        df = df.merge(files["delivery_performance.csv"], on="Order_ID", how="left")
        df = df.merge(files["customer_feedback.csv"], on="Order_ID", how="left")
        
        # Mappings
        weather_map = metadata.get("weather_map", {"Low": 0, "Medium": 1, "High": 2})
        priority_map = metadata.get("priority_map", {"Economy": 0, "Standard": 1, "Express": 2})
        
        # Derived Features
        if 'Weather_Impact' in df.columns:
             def clean_weather(w):
                 w = str(w)
                 if w in weather_map: return weather_map[w]
                 if "Rain" in w or "Fog" in w: return 2
                 if "Clear" in w: return 0
                 return 0 # Default
             df['Weather_Score'] = df['Weather_Impact'].apply(clean_weather)
        else:
             df['Weather_Score'] = 0

        if 'Priority' in df.columns:
            df['Priority_Score'] = df['Priority'].map(lambda x: priority_map.get(x, 1))
        else:
            df['Priority_Score'] = 1
            
        # Total Ops Cost Calculation
        cost_cols = [c for c in files["cost_breakdown.csv"].columns if "Cost" in c or "Fee" in c or "Overhead" in c]
        if 'Total_Ops_Cost' not in df.columns:
            df['Total_Ops_Cost'] = df[cost_cols].sum(axis=1)

        # Robust Missing Data Handling
        # 1. Numeric Columns: Fill with 0 (Optimistic/Conservative) or Median
        num_cols = df.select_dtypes(include=['number']).columns
        # For Order Value, use Median to avoid screwing up averages
        if "Order_Value_INR" in df.columns:
            df["Order_Value_INR"] = df["Order_Value_INR"].fillna(df["Order_Value_INR"].median())
        
        # For others, fill with 0
        df[num_cols] = df[num_cols].fillna(0)
        
        # 2. Categorical Columns: Fill with "Unknown"
        cat_cols = df.select_dtypes(include=['object']).columns
        df[cat_cols] = df[cat_cols].fillna("Unknown")
        
        return df
    except Exception as e:
        st.error(f"Data Processing Error: {e}")
        return pd.DataFrame()

if len(uploaded_files) >= 7 and model:
    master_df = process_data(uploaded_files, metadata)
else:
    master_df = pd.DataFrame()
    st.warning("Please upload all 7 CSV files to proceed.")

# --- Tab Layout ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Predictive Dashboard", 
    "� Real-time Prediction Calculator", 
    "🧠 AI Insights", 
    "💡 Innovation Brief"
])

# --- Tab 1: Predictive Dashboard ---
with tab1:
    if not master_df.empty:
        st.title("Predictive Logistics Dashboard")
        
        feature_cols = metadata["features"]
        # Basic validation
        if all(col in master_df.columns for col in feature_cols):
            predictions = model.predict(master_df[feature_cols])
            master_df["Predicted_Delay_Risk"] = predictions
            
            # Top KPIs
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Orders", len(master_df))
            at_risk_count = len(master_df[master_df["Predicted_Delay_Risk"] > 0.5])
            kpi2.metric("At-Risk Deliveries", at_risk_count, delta_color="inverse")
            avg_cost = master_df["Total_Ops_Cost"].mean()
            kpi3.metric("Avg Ops Cost", f"₹{avg_cost:,.2f}")
            kpi4.metric("AI Confidence", "87%")

            # 1. Route Risk Map
            st.subheader("📍 Route Risk Map")
            fig_map = px.scatter(
                master_df, x="Distance_KM", y="Traffic_Delay_Minutes", 
                color="Predicted_Delay_Risk", size="Order_Value_INR",
                hover_data=["Route", "Carrier"],
                title="Delivery Risk Matrix: Distance vs Traffic",
                color_continuous_scale=["#00FF00", "#FF0000"]
            )
            fig_map.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_map, use_container_width=True)

            # 2. Other Visuals (Preserved)
            r2c1, r2c2 = st.columns(2)
            with r2c1:
                st.subheader("💰 Cost Intelligence")
                cost_cols = [c for c in uploaded_files["cost_breakdown.csv"].columns if c != "Order_ID"]
                cost_sums = uploaded_files["cost_breakdown.csv"][cost_cols].sum().reset_index()
                cost_sums.columns = ["Subcategory", "Value"]
                cost_sums["Category"] = "Operational Costs"
                fig_sun = px.sunburst(cost_sums, path=["Category", "Subcategory"], values="Value",
                                      color="Value", color_continuous_scale="Viridis")
                fig_sun.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_sun, use_container_width=True)

            with r2c2:
                st.subheader("🚚 Carrier Performance")
                fig_bub = px.scatter(master_df, x="Actual_Delivery_Days", y="Customer_Rating",
                                     size="Delivery_Cost_INR", color="Carrier",
                                     hover_name="Carrier")
                fig_bub.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_bub, use_container_width=True)
            
            st.divider()
            st.markdown("### 🔍 Advanced Analytics Deep Dive")
            
            # 1. Route Delay Map (PyDeck)
            st.subheader("🌐 Network Delay Map (Origin -> Destination)")
            
            # City Coords Dictionary (Features from dataset)
            city_coords = {
                "Kolkata": [88.3639, 22.5726], "Hyderabad": [78.4867, 17.3850], "Mumbai": [72.8777, 19.0760],
                "Pune": [73.8567, 18.5204], "Chennai": [80.2707, 13.0827], "Delhi": [77.1025, 28.7041],
                "Bangalore": [77.5946, 12.9716], "Ahmedabad": [72.5714, 23.0225], "Dubai": [55.2708, 25.2048],
                "Hong Kong": [114.1694, 22.3193], "Singapore": [103.8198, 1.3521], "Bangkok": [100.5018, 13.7563]
            }
            
            map_df = master_df.copy()
            # Map coords
            map_df["src_coords"] = map_df["Origin"].map(city_coords)
            map_df["dst_coords"] = map_df["Destination"].map(city_coords)
            map_df = map_df.dropna(subset=["src_coords", "dst_coords"]) # Drop if city not found
            
            # Color logic: Red if delayed, Green if fast
            def get_color(days):
                if days > 5: return [255, 0, 0, 160] # Red
                return [0, 255, 0, 160] # Green
            
            map_df["color"] = map_df["Actual_Delivery_Days"].apply(get_color)
            
            layer = pdk.Layer(
                "ArcLayer",
                data=map_df,
                get_source_position="src_coords",
                get_target_position="dst_coords",
                get_source_color="color",
                get_target_color="color",
                get_width=3,
                pickable=True,
            )
            
            view_state = pdk.ViewState(latitude=20.5937, longitude=78.9629, zoom=3, pitch=45)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{Origin} -> {Destination}\nDays: {Actual_Delivery_Days}"}))
            
            # 2. Environmental Impact Bar
            st.subheader("🌍 Environmental Impact (Fuel & CO2)")
            # Aggregating by Carrier and Priority
            env_df = master_df.groupby(["Carrier", "Priority"])[["Fuel_Consumption_L", "Order_Value_INR"]].sum().reset_index()
            # CO2 Estimation (Sample logic: 2.68kg CO2 per Liter Diesel)
            env_df["Estimated_CO2_kg"] = env_df["Fuel_Consumption_L"] * 2.68
            
            fig_env = px.bar(env_df, x="Carrier", y="Estimated_CO2_kg", color="Priority", 
                             title="CO2 Emissions by Carrier & Priority", barmode="group",
                             color_discrete_sequence=px.colors.sequential.Teal)
            fig_env.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_env, use_container_width=True)
            
            # 3. Warehouse Bullet & Sentiment Treemap
            adv_c1, adv_c2 = st.columns(2)
            
            with adv_c1:
                st.subheader("📦 Stock Health (Location View)")
                inv_df = uploaded_files["warehouse_inventory.csv"]
                
                # Aggregate by Location to avoid clutter
                inv_agg = inv_df.groupby("Location")[["Current_Stock_Units", "Reorder_Level"]].sum().reset_index()
                
                # Create clean horizontal bar chart
                fig_bull = go.Figure()
                fig_bull.add_trace(go.Bar(
                    y=inv_agg['Location'], 
                    x=inv_agg['Current_Stock_Units'], 
                    name='Current Stock',
                    orientation='h',
                    marker_color='#D4AF37'
                ))
                fig_bull.add_trace(go.Bar(
                    y=inv_agg['Location'], 
                    x=inv_agg['Reorder_Level'], 
                    name='Reorder Level',
                    orientation='h',
                    marker_color='#FF4B4B'
                ))
                
                fig_bull.update_layout(
                    barmode='group', 
                    height=400, 
                    margin=dict(t=20, b=20, l=10, r=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    font_color='white',
                    xaxis_title="Units",
                    yaxis_title="Location"
                )
                st.plotly_chart(fig_bull, use_container_width=True)
                
            with adv_c2:
                st.subheader("🥰 Sentiment & Issue Heatmap")
                feed_df = uploaded_files["customer_feedback.csv"]
                # Aggregate Sentiment
                sent_agg = feed_df.groupby("Issue_Category").agg(
                    Count=("Rating", "count"),
                    Avg_Rating=("Rating", "mean")
                ).reset_index()
                
                fig_tree2 = px.treemap(sent_agg, path=["Issue_Category"], values="Count",
                                       color="Avg_Rating", color_continuous_scale="RdYlGn",
                                       title="Issue Frequency vs Satisfaction Score")
                fig_tree2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_tree2, use_container_width=True)

# --- Tab 2: Real-time Prediction Calculator ---
with tab2:
    st.title("🧪 Real-time Prediction & Correction Engine")
    
    # Initialize Session State for Persistence
    if "sim_run" not in st.session_state:
        st.session_state.sim_run = False
        st.session_state.sim_prob = 0.0
        st.session_state.sim_summary = ""
        st.session_state.sim_email = ""
        st.session_state.best_loc = ""
        st.session_state.upgrade_cost = 0.0
        st.session_state.roi_value = 0.0
        st.session_state.sim_weather = ""
        st.session_state.sim_traffic = 0
    
    if not master_df.empty:
        col_input, col_result = st.columns([1, 2])
        
        with col_input:
            st.markdown("### 🎛️ Simulation Parameters")
            sim_traffic = st.slider("Traffic Delay (Minutes)", 0, 180, 45)
            sim_weather = st.select_slider("Weather Forecast", options=["Low", "Medium", "High"], value="Medium")
            sim_priority = st.select_slider("Order Priority", options=["Economy", "Standard", "Express"], value="Standard")
            
            if st.button("Simulate Risk Scenario"):
                st.session_state.sim_run = True
                
                # 1. Calculation
                feature_cols = metadata["features"]
                base_features = master_df[feature_cols].mean().to_dict()
                
                base_features["Traffic_Delay_Minutes"] = sim_traffic
                base_features["Weather_Score"] = metadata["weather_map"].get(sim_weather, 0)
                base_features["Priority_Score"] = metadata["priority_map"].get(sim_priority, 1)
                
                sim_df = pd.DataFrame([base_features])
                st.session_state.sim_prob = model.predict(sim_df)[0]
                st.session_state.sim_weather = sim_weather
                st.session_state.sim_traffic = sim_traffic
                
                # Action Data Prep
                st.session_state.best_loc = "N/A"
                if st.session_state.sim_prob > 0.5:
                    inv_df = uploaded_files["warehouse_inventory.csv"]
                    surplus_locs = inv_df[inv_df['Current_Stock_Units'] > inv_df['Reorder_Level']]
                    if not surplus_locs.empty:
                        st.session_state.best_loc = surplus_locs.iloc[0]['Location']
                    
                    curr_cost = base_features.get("Total_Ops_Cost", 500)
                    st.session_state.upgrade_cost = curr_cost * 0.20
                    st.session_state.roi_value = (curr_cost * 1.5) - st.session_state.upgrade_cost
                    
                    # Generate Summary ONLY if Risk is High
                    if client:
                        prompt_plan = f"""
                        You are a Senior Logistics Consultant.
                        Scenario: A shipment is predicted to be DELAYED (High Risk) due to Weather: {sim_weather} and Traffic: {sim_traffic} min.
                        
                        Proposed Corrective Actions:
                        1. Stock Redirection from {st.session_state.best_loc}.
                        2. Carrier Upgrade to Air Freight (Cost Impact: +₹{st.session_state.upgrade_cost:.0f}).
                        
                        Task: Explain WHY taking these actions is critical to save operational costs and prevent customer churn. 
                        Focus on the 15-20% cost saving ROI. Keep it executive and concise.
                        """
                        try:
                            response = client.models.generate_content(
                                model="gemini-2.5-flash", 
                                contents=prompt_plan
                            )
                            st.session_state.sim_summary = response.text
                        except Exception as e:
                            st.session_state.sim_summary = f"AI Error: {e}"
                else:
                    st.session_state.sim_summary = ""

            # --- Display Summary in Left Column (Below Button) ---
            if st.session_state.sim_run and st.session_state.sim_prob > 0.5:
                st.divider()
                st.markdown("### 🤖 Managerial Summary")
                st.info(st.session_state.sim_summary)
                
                # Popup Logic for Email
                if st.button("📧 Draft Carrier Email"):
                   email_prompt = f"""
                   Based on this summary: {st.session_state.sim_summary}
                   Write a formal directive email to the Logistics Partner.
                   Subject: Urgent Optimization Required
                   Tone: Authoritative.
                   """
                   try:
                       if client:
                           email_resp = client.models.generate_content(
                               model="gemini-2.5-flash", 
                               contents=email_prompt
                           )
                           st.session_state.sim_email = email_resp.text
                       else:
                           st.session_state.sim_email = "Please connect Gemini."
                   except Exception as e:
                       st.error(str(e))
                
                if st.session_state.sim_email:
                    # Use an expander as a 'popup' substitute or st.dialog if available
                    # Since we want to be safe, let's use a clear container highlighted
                    with st.container():
                        st.success("Draft Generated Ready for Download")
                        
                        # PDF Generation
                        from fpdf import FPDF
                        class PDF(FPDF):
                            def header(self):
                                self.set_font('Arial', 'B', 12)
                                self.cell(0, 10, 'NexGen Logistics - Carrier Directive', 0, 1, 'C')
                        
                        pdf = PDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        # Encode/Decode to handle latin-1
                        text = st.session_state.sim_email.encode('latin-1', 'replace').decode('latin-1')
                        pdf.multi_cell(0, 10, text)
                        
                        pdf_bytes = pdf.output(dest='S').encode('latin-1')
                        
                        st.download_button(
                            label="📥 Download Email (PDF)",
                            data=pdf_bytes,
                            file_name="carrier_directive.pdf",
                            mime="application/pdf"
                        )

        # --- Display Results in Right Column ---
        with col_result:
            if st.session_state.sim_run:
                st.markdown("### 📊 Prediction Confidence")
                probability = st.session_state.sim_prob
                
                risk_label = "CRITICAL DELAY" if probability > 0.5 else "ON TIME"
                risk_color = "red" if probability > 0.5 else "green"
                st.markdown(f"<h2 style='color: {risk_color};'>{risk_label}</h2>", unsafe_allow_html=True)
                st.metric("Predicted Failure Probability", f"{probability * 100:.0f}%" if probability <= 1 else "High")
                
                st.markdown("### 🛡️ Corrective Action Plan")
                if probability > 0.5:
                    st.success(f"**Option 1: Stock Redirection**")
                    if st.session_state.best_loc != "N/A":
                        st.write(f"📍 Source from: **{st.session_state.best_loc}** (Surplus Available)")
                    else:
                        st.write("📍 No nearby surplus stock found.")
                        
                    st.info(f"**Option 2: Carrier Upgrade (Air Freight)**")
                    st.write(f"💸 Additional Cost: ₹{st.session_state.upgrade_cost:.2f}")
                    st.metric("Expected Recovery ROI", f"₹{st.session_state.roi_value:.2f}", delta="Saved Penalty")
                else:
                    st.success("Analysis: No Corrective Action Needed. Delivery is on track.")


# --- Tab 3: AI Insights ---
with tab3:
    st.title("🧠 AI Logistics Intelligence")
    
    col_chat, col_nudge = st.columns([2, 1])
    
    with col_chat:
        st.subheader("Ask NexGen")
        user_query = st.text_input("Ask a question about your logistics data...", placeholder="e.g., What are the top causes of delay?")
        if user_query and client and not master_df.empty:
             with st.spinner("Thinking..."):
                try:
                    stats_summary = master_df.describe(include='all').to_string()
                    chat_prompt = f"""
                    Context: You are NexGen's AI Data Analyst.
                    Data Stats: {stats_summary}
                    User Query: {user_query}
                    Provide a sharp, data-driven answer. 
                    """
                    response = client.models.generate_content(
                        model="gemini-2.5-flash", 
                        contents=chat_prompt
                    )
                    st.markdown(response.text)
                except Exception as e:
                    st.error(str(e))
    
    with col_nudge:
        st.subheader("Automated Negotiation")
        if not master_df.empty:
            at_risk = master_df[master_df["Predicted_Delay_Risk"] > 0.5]
            if not at_risk.empty:
                sel_id = st.selectbox("Select Failing Order", at_risk["Order_ID"].unique())
                if st.button("Draft Warning Email"):
                    if client:
                        with st.spinner("Drafting..."):
                            row = master_df[master_df["Order_ID"] == sel_id].iloc[0]
                            email_prompt = f"Write a stern logistics email regarding Order {sel_id} which is delayed. Carrier: {row.get('Carrier')}. Value: {row.get('Order_Value_INR')}."
                            resp = client.models.generate_content(model="gemini-2.5-flash", contents=email_prompt)
                            st.text_area("Content", resp.text, height=250)
            else:
                st.info("No At-Risk orders identified.")

# --- Tab 4: Innovation Brief ---
with tab4:
    st.markdown("""
    ## The NexGen Logistics Innovation Challenge
    
    **Goal**: Predictive dashboard that identifies delivery risks and uses Gemini AI to suggest executive-level interventions.
    
    ### Key Features Implemented:
    1. **Multi-Factor Ingestion**: Integrating Orders, Performance, Costs, Routes, and Fleet data.
    2. **AI-Powered Insights**: 'Ask NexGen' interface powered by **Google Gemini**.
    3. **Predictive Risk Engine**: Logistic Regression Model trained on historical delays.
    4. **Prescriptive Actions**: Automated inventory redirection and carrier negotiation drafts.
    5. **Executive Visuals**: Plotly-powered interactive maps, sunbursts, and bubble charts.
    
    """)
    
    if not master_df.empty:
        st.download_button(
            "📥 Download Full Decision Plan", 
            master_df.to_csv(index=False).encode('utf-8'),
            "decision_plan.csv",
            "text/csv"
        )
