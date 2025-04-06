import streamlit as st
import requests
import re
import datetime
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import random


st.set_page_config(page_title="AI Climate Impact Calculator", layout="centered")


summarizer = pipeline("summarization")
flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


def extract_quantities(text):
   distance = 0
   electricity = 0
   hours = 0


   match_distance = re.search(r'(\d+(\.\d+)?)\s*(km|kilometers?|miles?)', text.lower())
   if match_distance:
       distance = float(match_distance.group(1)) * (1.609 if 'mile' in match_distance.group(3) else 1)


   match_kwh = re.search(r'(\d+(\.\d+)?)\s*(kwh|kilowatt-hours?)', text.lower())
   if match_kwh:
       electricity = float(match_kwh.group(1))


   match_hours = re.search(r'(\d+(\.\d+)?)\s*(hours?|hrs?)', text.lower())
   if match_hours:
       hours = float(match_hours.group(1))


   return distance, electricity, hours


def get_sustainability_tip(text):
   text = text.lower()
   if "car" in text:
       return "🚲 Consider biking or using public transport once a week to cut down emissions."
   elif "meat" in text or any(x in text for x in ["chicken", "beef", "pork", "lamb"]):
       return "🥗 Try a meat-free day — it can reduce your footprint by up to 25%."
   elif "ac" in text or "heater" in text:
       return "🌡️ Optimize appliance use. Set thermostats wisely and reduce unnecessary electricity usage."
   else:
       return "🌱 Small changes make big impact. Keep tracking to stay sustainable!"


def store_history(date, value):
   if "history" not in st.session_state:
       st.session_state.history = {}
   st.session_state.history[str(date)] = value


def plot_history():
   if "history" in st.session_state:
       dates = list(st.session_state.history.keys())
       values = list(st.session_state.history.values())
       plt.figure(figsize=(6, 2))
       plt.plot(dates, values, marker='o')
       plt.xticks(rotation=45)
       plt.title("Daily Carbon Footprint (kg CO₂)")
       st.pyplot(plt.gcf())


def ask_ai_routine_recommendation(user_text, distance_km, electricity_kwh, meat_emission):
   prompt = (
       f"Suggest 2-3 realistic, eco-friendly changes the user can try tomorrow to reduce carbon emissions.\n"
       f"Routine summary: {user_text}\n"
       f"Transport distance: {distance_km:.1f} km, Electricity used: {electricity_kwh:.1f} kWh, Meat-related emissions: {meat_emission:.1f} kg CO2."
   )
   try:
       inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True)
       outputs = flan_model.generate(**inputs, max_new_tokens=120)
       result = flan_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
       sentences = result.split(". ")
       filtered = []
       for s in sentences:
           if s not in filtered and len(s.strip()) > 5:
               filtered.append(s)
       return ". ".join(filtered).strip()
   except Exception as e:
       return f"Error generating recommendation: {str(e)}"


def ask_ai(question):
   prompt = f"""
You are a climate science expert assistant. Answer the following question with a helpful, intelligent explanation.


Include:
- A simple definition
- Why it matters for the environment
- One example or real-world impact


Question: {question}
Answer:
"""
   inputs = flan_tokenizer(prompt, return_tensors="pt")
   outputs = flan_model.generate(**inputs, max_new_tokens=300)
   response = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
   return response.strip()


def generate_detailed_summary(user_text, activities, distance_km, electricity_kwh, meat_emission, total_emission, goal, points, tip):
   summary = (
       f"Today, based on your input:\n"
       f"- Transportation was a major part of your routine with {distance_km:.2f} km traveled, causing approximately {distance_km * 0.06:.2f} kg CO₂ emissions.\n"
       f"- You used {electricity_kwh:.2f} kWh of electricity, emitting {electricity_kwh * 0.4:.2f} kg CO₂.\n"
       f"- Meat consumption contributed an estimated {meat_emission:.2f} kg CO₂.\n\n"
       f"In total, your carbon footprint was **{total_emission:.2f} kg CO₂**.\n"
       f"Your impact score was **{points} pts**, and you saved approximately {points // 50} trees' worth of CO₂ today.\n"
   )
   if total_emission <= goal:
       summary += f"🎉 You met your daily goal of {goal:.2f} kg CO₂!\n"
   else:
       summary += f"⚠️ You exceeded your goal of {goal:.2f} kg CO₂.\n"
   summary += f"\nAI Tip: {tip}"
   return summary


def get_streak_message():
   if "history" not in st.session_state or len(st.session_state.history) < 2:
       return "Start tracking to build your streak!"
   dates = list(st.session_state.history.keys())
   vals = list(st.session_state.history.values())
   if vals[-1] < vals[-2]:
       return "🔥 Great job! You emitted less CO₂ than yesterday!"
   return "📉 Try reducing meat or transport to cut emissions tomorrow."


st.title("🌍 AI Climate Impact Calculator")
st.write("Describe your daily routine and get an AI-powered estimate of your carbon footprint.")


st.header("🗣️ What did you do today?")
user_input = st.text_area("", placeholder="Type your day’s activities here…")


goal = st.number_input("🎯 Set a carbon footprint goal (kg CO₂)", min_value=1.0, max_value=20.0, value=5.0)


if st.button("Calculate Footprint"):
   with st.spinner("🔍 Using AI to analyze your routine..."):
       classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
       labels = ["transportation", "electricity usage", "meat consumption"]
       result = classifier(user_input, candidate_labels=labels)
       activities = {label: score for label, score in zip(result["labels"], result["scores"])}


       st.subheader("🧠 Detected Activities:")
       for k, v in activities.items():
           st.write(f"{k.title()}: {v * 100:.1f}% confidence")


       distance_km, electricity_kwh, hours_used = extract_quantities(user_input)
       if electricity_kwh == 0 and hours_used > 0:
           electricity_kwh = hours_used * 0.4


       carbon_intensity = 400
       st.warning("⚠️ Could not fetch real-time carbon intensity. Using fallback value (400g/kWh).")


       transport_emission = distance_km * 0.06
       electricity_emission = electricity_kwh * (carbon_intensity / 1000)
       meat_emission = 2.5 if activities["meat consumption"] > 0.4 else 0


       total = transport_emission + electricity_emission + meat_emission
       today = datetime.date.today()
       store_history(today, total)


       st.subheader("📊 Estimated Carbon Footprint:")
       st.write(f"🚗 Transport: **{transport_emission:.2f} kg CO₂** ({distance_km} km)")
       st.write(f"⚡ Electricity: **{electricity_emission:.2f} kg CO₂** ({electricity_kwh:.2f} kWh, {carbon_intensity} gCO₂/kWh)")
       st.write(f"🍖 Meat: **{meat_emission:.2f} kg CO₂**")
       st.success(f"🌱 Total Estimated Footprint: **{total:.2f} kg CO₂**")


       points = max(0, round((10 - total) * 10))
       trees_saved = points // 50
       st.markdown(f"💥 Impact Score: **{points} pts** | 🌳 Trees Saved Equivalent: **{trees_saved}**")


       if total <= goal:
           st.success(f"🎉 You met your goal of {goal} kg CO₂ today!")
       else:
           over = total - goal
           st.warning(f"⚠️ You exceeded your goal by {over:.2f} kg CO₂. Try adjusting your routine tomorrow.")


       tip = get_sustainability_tip(user_input)
       st.info(tip)


       st.download_button("📤 Export Report", f"Daily CO2 Footprint: {total:.2f} kg CO2\nScore: {points} pts\nTip: {tip}", file_name="footprint_report.txt")


       st.subheader("📈 Your Daily Progress")
       plot_history()
       st.caption(get_streak_message())


       st.subheader("🧠 AI Summary of Your Routine")
       st.write(generate_detailed_summary(user_input, activities, distance_km, electricity_kwh, meat_emission, total, goal, points, tip))


       st.subheader("🧠 AI Routine Recommender")
       recommendation = ask_ai_routine_recommendation(user_input, distance_km, electricity_kwh, meat_emission)
       st.write(recommendation)


       st.subheader("🏆 Community Leaderboard")
       leaderboard = sorted([random.randint(50, 120) for _ in range(10)] + [points], reverse=True)
       for i, p in enumerate(leaderboard[:5], 1):
           highlight = "👉 You" if p == points else ""
           st.write(f"{i}. Score: {p} pts {highlight}")


       st.caption("🧠 Powered by AI. Emissions calculated using extracted quantities and activity detection.")


# Optional Advanced UI for Manual Inputs (Inspired by Screenshot Design)
st.markdown("---")
st.subheader("📂 Manual Impact Explorer")
tabs = st.tabs(["Transportation", "Housing", "Food", "Consumption"])

with tabs[0]:
    st.markdown("<h3 style='font-weight:600;'>🚗 Transportation Impact</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        miles = st.slider("Miles Driven Per Year", 0, 30000, 5000, step=100)
    with col2:
        st.markdown(f"<p style='text-align:right;'>{miles:,} miles</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        mpg = st.slider("Fuel Efficiency (MPG)", 5, 100, 25)
    with col2:
        st.markdown(f"<p style='text-align:right;'>{mpg} mpg</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        flight_miles = st.slider("Flight Miles Per Year", 0, 20000, 1000, step=100)
    with col2:
        st.markdown(f"<p style='text-align:right;'>{flight_miles:,} miles</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        public_transit = st.slider("Public Transportation Usage (% of trips)", 0, 100, 10)
    with col2:
        st.markdown(f"<p style='text-align:right;'>{public_transit}%</p>", unsafe_allow_html=True)

    st.markdown("""
        <style>
        div.stSlider > label {
            font-weight: 500;
            font-size: 0.9rem;
            margin-bottom: 0.3rem;
        }
        div.stButton > button {
            background-color: #2ecc71;
            color: white;
            font-weight: bold;
            padding: 0.75em 2em;
            border-radius: 50px;
            border: none;
            font-size: 16px;
            margin-top: 1.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🔁 Recalculate Transport Footprint", key="recalculate_button"):

        car_emission = (miles / mpg) * 8.89 / 1000
        air_emission = flight_miles * 0.15 / 1000
        saved = car_emission * (public_transit / 100)
        total = round(car_emission + air_emission - saved, 2)

        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.markdown("#### Your Carbon Footprint")
            st.markdown(f"<h1 style='color:#2ecc71; font-size: 48px;'>{total}</h1><span style='font-size: 20px;'>tons CO₂e/year</span>", unsafe_allow_html=True)

        

        st.markdown("#### Comparison to Targets")
        st.progress(min(total / 16, 1.0))
        st.markdown(f"Compared to National Avg (16 tons): **{round((total / 16) * 100)}%**")

        st.progress(min(total / 2, 1.0), text="Sustainable Target")
        st.markdown(f"Sustainable Target (2 tons): **{round((total / 2) * 100)}%**")

        st.markdown("#### Recommendations:")
        st.markdown("- ✅ Look into renewable energy options for your home to reduce emissions.")
        st.markdown("- ✅ Increase your recycling efforts to reduce waste-related emissions.")

with tabs[1]:
    energy_kwh = st.slider("Monthly Energy Usage (kWh)", 0, 2000, 500)
    renewable_pct = st.slider("Renewable Energy Percentage", 0, 100, 10)
    home_size = st.slider("Home Size (sq ft)", 100, 10000, 1000)
    efficiency_rating = st.slider("Home Efficiency Rating (%)", 0, 100, 50)

    if st.button("🏠 Recalculate Housing Footprint", key="housing_recalc"):
        non_renewable_kwh = energy_kwh * (1 - renewable_pct / 100)
        housing_energy_emission = non_renewable_kwh * 0.4  # kg/month
        housing_size_emission = (home_size / 1000) * (1 - efficiency_rating / 100) * 1.5  # kg/month
        total_kg = housing_energy_emission + housing_size_emission
        total_tons = round((total_kg * 12) / 1000, 2)

        # ... rest of the output ...

        st.markdown("#### Your Carbon Footprint")
        st.markdown(f"<h1 style='color:#3498db; font-size: 48px;'>{total_tons}</h1><span style='font-size: 20px;'>tons CO₂e/year</span>", unsafe_allow_html=True)

        st.markdown("#### Comparison to Targets")
        st.progress(min(total_tons / 16, 1.0))
        st.markdown(f"Compared to National Avg (16 tons): **{round((total_tons / 16) * 100)}%**")

        st.progress(min(total_tons / 2, 1.0), text="Sustainable Target")
        st.markdown(f"Sustainable Target (2 tons): **{round((total_tons / 2) * 100)}%**")

        st.markdown("#### Recommendations:")
        st.markdown("- 🏠 Improve insulation and upgrade to energy-efficient appliances.")
        st.markdown("- 🌞 Increase renewable energy usage (solar, wind).")



with tabs[2]:
    meat_servings = st.slider("Meat Consumption (Servings Per Week)", 0, 21, 3)
    local_pct = st.slider("Local Food Percentage", 0, 100, 20)
    food_waste_pct = st.slider("Food Waste Percentage", 0, 100, 15)

    if st.button("🍽️ Recalculate Food Footprint", key="food_recalc"):
        meat_emission = meat_servings * 1.8
        non_local = (1 - local_pct / 100) * 0.5
        waste = food_waste_pct / 100 * 2
        total_kg_per_week = meat_emission + non_local + waste
        total_tons = round((total_kg_per_week * 52) / 1000, 2)

        st.markdown("#### Your Carbon Footprint")
        st.markdown(f"<h1 style='color:#e67e22; font-size: 48px;'>{total_tons}</h1><span style='font-size: 20px;'>tons CO₂e/year</span>", unsafe_allow_html=True)

        st.markdown("#### Comparison to Targets")
        st.progress(min(total_tons / 16, 1.0))
        st.markdown(f"Compared to National Avg (16 tons): **{round((total_tons / 16) * 100)}%**")

        st.progress(min(total_tons / 2, 1.0), text="Sustainable Target")
        st.markdown(f"Sustainable Target (2 tons): **{round((total_tons / 2) * 100)}%**")

        st.markdown("#### Recommendations:")
        st.markdown("- 🥗 Try meatless meals a few times a week.")
        st.markdown("- 🌽 Prioritize local, seasonal produce.")
        st.markdown("- 🗑️ Reduce food waste by meal planning.")


with tabs[3]:
    monthly_spending = st.slider("Monthly Spending ($)", 0, 2000, 500)
    recycling_rate = st.slider("Recycling Rate (%)", 0, 100, 25)
    sustainable_pct = st.slider("Sustainable Products (%)", 0, 100, 20)

    if st.button("🛍️ Recalculate Consumption Footprint", key="consumption_recalc"):
        base = monthly_spending * 0.02
        saved = (base * (recycling_rate / 100) * 0.3) + (base * (sustainable_pct / 100) * 0.25)
        net_monthly_kg = base - saved
        total_tons = round((net_monthly_kg * 12) / 1000, 2)

        st.markdown("#### Your Carbon Footprint")
        st.markdown(f"<h1 style='color:#9b59b6; font-size: 48px;'>{total_tons}</h1><span style='font-size: 20px;'>tons CO₂e/year</span>", unsafe_allow_html=True)

        st.markdown("#### Comparison to Targets")
        st.progress(min(total_tons / 16, 1.0))
        st.markdown(f"Compared to National Avg (16 tons): **{round((total_tons / 16) * 100)}%**")

        st.progress(min(total_tons / 2, 1.0), text="Sustainable Target")
        st.markdown(f"Sustainable Target (2 tons): **{round((total_tons / 2) * 100)}%**")

        st.markdown("#### Recommendations:")
        st.markdown("- 🔄 Buy less, buy better — focus on quality and durability.")
        st.markdown("- ♻️ Choose products with recyclable packaging.")

st.markdown("---")
st.subheader("🧮 Total Carbon Footprint Calculator")

if st.button("Calculate Total Footprint", key="total_footprint_btn"):
    # TRANSPORT
    car_emission = (miles / mpg) * 8.89 / 1000  # kg CO₂
    air_emission = flight_miles * 0.15 / 1000   # kg CO₂
    saved = car_emission * (public_transit / 100)
    transport_total = round(car_emission + air_emission - saved, 2)

    # HOUSING
    non_renewable_kwh = energy_kwh * (1 - renewable_pct / 100)
    housing_energy_emission = non_renewable_kwh * 0.4
    housing_size_emission = (home_size / 1000) * (1 - efficiency_rating / 100) * 1.5
    housing_total = round(housing_energy_emission + housing_size_emission, 2)

    # FOOD
    food_meat_emission = meat_servings * 1.8
    food_nonlocal = (1 - local_pct / 100) * 0.5
    food_waste = food_waste_pct / 100 * 2
    food_total = round(food_meat_emission + food_nonlocal + food_waste, 2)

    # CONSUMPTION
    base_consumption = monthly_spending * 0.02
    recycle_save = base_consumption * (recycling_rate / 100) * 0.3
    sustainable_save = base_consumption * (sustainable_pct / 100) * 0.25
    consumption_total = round(base_consumption - recycle_save - sustainable_save, 2)

    # TOTAL
    grand_total = round(transport_total + housing_total + food_total + consumption_total, 2)
    st.session_state["grand_total"] = grand_total


    st.success(f"🌍 Your Total Estimated Carbon Footprint: **{grand_total} kg CO₂/month**")

    labels = ['Transportation', 'Housing', 'Food', 'Consumption']
    values = [transport_total, housing_total, food_total, consumption_total]
    colors = ['#1abc9c', '#3498db', '#e67e22', '#9b59b6']

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        startangle=90,
        autopct='%1.1f%%',
        wedgeprops=dict(width=0.4)
    )
    ax.axis('equal')
    st.pyplot(fig)

    st.markdown("✅ Tip: Focus on the largest slice to maximize impact reduction.")


    if st.button("Calculate My Footprint", key="consumption_calc"):
        base_emission = monthly_spending * 0.02  # kg CO2e per $ spent (example factor)
        recycling_savings = base_emission * (recycling_rate / 100) * 0.3
        sustainable_savings = base_emission * (sustainable_pct / 100) * 0.25

        net_emission = round(base_emission - recycling_savings - sustainable_savings, 2)

        st.success(f"🛒 Estimated Consumption Emissions: **{net_emission} kg CO₂/month**")
        st.markdown("- 💰 Lower spending = less consumption = less footprint.")
        st.markdown("- ♻️ Recycling and sustainable choices reduce overall impact.")
        # === IMPACT TIMELINE FEATURE ===
if "grand_total" in st.session_state:
    # Show toggle button only after footprint is calculated
    if "show_timeline" not in st.session_state:
        st.session_state.show_timeline = False

    st.toggle("📅 Impact Timeline", key="show_timeline", value=st.session_state.show_timeline)

    if st.session_state.show_timeline:
        st.subheader("📈 Your Emission Reduction Timeline")

        years = st.slider("Timeline (years)", 1, 10, 5)
        reduction_rate = st.slider("Annual Reduction Rate (%)", 0, 20, 7)

        start_year = datetime.datetime.now().year
        starting_emissions = st.session_state["grand_total"] / 1000  # Convert kg to tons

        future_years = list(range(start_year, start_year + years + 1))
        future_emissions = [
            round(starting_emissions * ((1 - reduction_rate / 100) ** i), 2)
            for i in range(years + 1)
        ]

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.fill_between(future_years, future_emissions, color='lightgreen', alpha=0.7)
        ax.plot(future_years, future_emissions, marker='o', color='green')

        for i, val in enumerate(future_emissions):
            if i % 2 == 0 or i == len(future_emissions) - 1:
                ax.annotate(f"{val} t", (future_years[i], future_emissions[i] + 0.2), fontsize=8)

        ax.set_ylim(0, max(future_emissions) + 2)
        ax.set_xlim(future_years[0], future_years[-1])
        ax.set_ylabel("Tons CO₂e")
        ax.set_title("Your Carbon Footprint Over Time")
        ax.grid(visible=True, linestyle="--", alpha=0.4)
        st.pyplot(fig)

        st.markdown(f"📍 **Year {start_year}**: {starting_emissions:.2f} tons CO₂e")

    import plotly.graph_objects as go

# === HISTORICAL EMISSIONS COMPARISON ===
if "grand_total" in st.session_state:
    st.subheader("📊 Historical Emissions Comparison")
    st.markdown("See how global, US, and EU emissions have changed over time compared to your footprint.")

    # Historical data
    years = list(range(1996, 2026))

    # Simulated emissions trends
    global_avg = [round(4.2 + 0.02 * (i % 10), 1) for i in range(len(years))]       # Fluctuating upward
    us_avg = [round(20 - 0.2 * i, 1) for i in range(len(years))]                    # Steady decline
    eu_avg = [round(10.5 - 0.1 * i, 1) for i in range(len(years))]                  # Gentle drop

    # Your latest footprint in tons (converted from kg)
    your_footprint_tons = round(st.session_state["grand_total"] / 1000, 2)

    # Create plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=years, y=global_avg,
        mode='lines+markers',
        name='🌍 Global Average',
        line=dict(color='mediumpurple')
    ))

    fig.add_trace(go.Scatter(
        x=years, y=us_avg,
        mode='lines+markers',
        name='🇺🇸 US Average',
        line=dict(color='mediumseagreen')
    ))

    fig.add_trace(go.Scatter(
        x=years, y=eu_avg,
        mode='lines+markers',
        name='🇪🇺 EU Average',
        line=dict(color='orange')
    ))

    fig.add_trace(go.Scatter(
        x=[2025], y=[your_footprint_tons],
        mode='markers+text',
        name='🟥 Your Footprint',
        marker=dict(color='red', size=10),
        text=[f"{your_footprint_tons} t"],
        textposition='top right'
    ))

    fig.update_layout(
        title="Your Carbon Footprint vs Historical Emissions",
        xaxis_title="Year",
        yaxis_title="Tons CO₂e per person",
        hovermode="x unified",
        height=400,
        legend=dict(x=0, y=1.15, orientation="h"),
        margin=dict(t=60, l=40, r=40, b=40),
        plot_bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)


st.subheader("💬 Ask the AI Eco Assistant")
ai_query = st.text_input("Type your question (e.g., 'How can I reduce emissions while commuting?')")
if st.button("Get AI Advice") and ai_query:
   st.write(ask_ai(ai_query))
   # 🔍 Informative Block about Carbon Footprint
st.markdown("---")
st.header("What is a Carbon Footprint?")
st.write("""
A carbon footprint is the total amount of greenhouse gases (including carbon dioxide and methane) that are generated by our actions.
The average carbon footprint for a person in the United States is 16 tons, one of the highest rates in the world.

Globally, the average carbon footprint is closer to 4 tons. To have the best chance of avoiding a 2ºC rise in global temperatures,
the average global carbon footprint per person needs to drop to under 2 tons by 2050.
""")
st.markdown("### Main Sources of Carbon Emissions")
st.markdown("- 🟢 **Transportation (28%)** — Cars, planes, ships, and trains")
st.markdown("- 🔵 **Electricity production (27%)** — Burning fossil fuels")
st.markdown("- 🟠 **Industry (22%)** — Manufacturing and production")
st.markdown("- 🟤 **Commercial and residential (12%)** — Heating, cooling")
st.markdown("- 🔴 **Agriculture (10%)** — Crop and livestock production")

st.markdown("- 🔴 **Agriculture (10%)** — Crop and livestock production")



