import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle
from datetime import date
import difflib

st.set_page_config(page_title="Nomadiq | Smart Flight Predictor", layout="centered")

VALID_AIRPORTS = {
"DEL","BOM","BLR","HYD","MAA","CCU","PNQ","AMD","GOI","COK",
"JAI","LKO","PAT","SXR","ATQ","IXB","IXU","IDR","TRV",
"IXZ","IXM","IXL","DIB","SHL","JLR","RAJ","RPR","VNS","AYJ",
"GAY","CJB","GWL","UDR","IMF","BHO","JDH","AJL","KUU",
"IXE","IXJ","GAU","BBI","IXR","NAG","CIM","IXC"
}


ML_ROUTES = {
("DEL","BOM"),("BLR","DEL"),("BLR","BOM"),("HYD","DEL"),("HYD","BOM"),
("MAA","DEL"),("MAA","BOM"),("CCU","DEL"),("CCU","BOM"),
("PNQ","DEL"),("PNQ","BLR"),("AMD","BOM"),("AMD","DEL"),
("GOI","BOM"),("GOI","BLR"),("COK","BLR"),
("JAI","DEL"),("LKO","DEL"),("PAT","DEL"),("SXR","DEL"),
("BOM","DEL"),("DEL","BLR"),("BOM","BLR"),
("DEL","HYD"),("BOM","HYD"),
("DEL","MAA"),("BOM","MAA"),
("DEL","CCU"),("BOM","CCU"),
("DEL","PNQ"),("BLR","PNQ"),
("BOM","AMD"),("DEL","AMD"),
("BOM","GOI"),("BLR","GOI"),
("BLR","COK"),
("DEL","JAI"),("DEL","LKO"),("DEL","PAT"),("DEL","SXR"),
}


bst = xgb.Booster()
bst.load_model("xgb_flight_model.json")

with open("model_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

features = artifacts["features"]
freq_maps = artifacts["freq_maps"]
seat_median = artifacts["seat_median"]


def fallback_prediction(current_price, days_left):

    if days_left >= 45:
        discount = 0.11
    elif days_left >= 30:
        discount = 0.08
    elif days_left >= 20:
        discount = 0.06
    elif days_left >= 15:
        discount = 0.04
    elif days_left >= 10:
        discount = 0.02
    else:
        discount = 0.01

    return round(current_price * (1 - discount), 2)


def suggest_airport(code):
    suggestion = difflib.get_close_matches(code, VALID_AIRPORTS, n=1, cutoff=0.6)
    return suggestion[0] if suggestion else None


st.title("NomadiQ ")
st.caption("Flight Fare Prediction System")

col1, col2 = st.columns(2)

with col1:
    origin = st.text_input("Origin (IATA Code)", "DEL").upper()
    departure_date = st.date_input("Departure Date", min_value=date.today())
    departure_time = st.time_input("Departure Time")
    airline = st.selectbox("Airline Name",
                           ["Indigo", "Air India", "Vistara", "SpiceJet", "Akasa Air"])

with col2:
    destination = st.text_input("Destination (IATA Code)", "BOM").upper()
    arrival_time = st.time_input("Arrival Time")
    cabin_class = st.selectbox("Cabin Class", ["Economy", "Business"])

current_price = st.number_input(
    "Current Price (₹)",
    min_value=0.0,
    step=100.0,
    value=None,
    placeholder="Enter current price"
)

today = date.today()
days_left = (departure_date - today).days
st.info(f"📅 Days Before Departure: {days_left} days")


if st.button("Predict Future Fare"):

    if days_left < 5:
        st.error("❌ Minimum 5 days required before departure.")
        st.stop()

  
    if len(origin) != 3 or not origin.isalpha():
        st.error("❌ Origin must be a 3-letter airport code.")
        st.stop()

    if len(destination) != 3 or not destination.isalpha():
        st.error("❌ Destination must be a 3-letter airport code.")
        st.stop()


    if origin not in VALID_AIRPORTS:
        suggestion = suggest_airport(origin)
        if suggestion:
            st.error(f"❌ '{origin}' not found. Did you mean '{suggestion}'?")
        else:
            st.error(f"❌ '{origin}' is not a valid airport code.")
        st.stop()

    if destination not in VALID_AIRPORTS:
        suggestion = suggest_airport(destination)
        if suggestion:
            st.error(f"❌ '{destination}' not found. Did you mean '{suggestion}'?")
        else:
            st.error(f"❌ '{destination}' is not a valid airport code.")
        st.stop()

    route = (origin, destination)

   
    if route in ML_ROUTES:

        df = pd.DataFrame([{
            "origin": origin,
            "destination": destination,
            "departure_date": str(departure_date),
            "departure_time": str(departure_time),
            "arrival_time": str(arrival_time),
            "min_price": current_price,
            "days_before_departure": days_left,
            "cabin_class": cabin_class,
            "airline_name": airline,
            "fare_type": "Saver",
            "checkin_baggage": "15 Kg",
            "cabin_baggage": "7 KG"
        }])

        dep_date = pd.to_datetime(df["departure_date"], errors="coerce")
        df["dep_weekday"] = dep_date.dt.weekday
        df["dep_month"] = dep_date.dt.month

        def split_time(col):
            t = pd.to_datetime(df[col], errors="coerce")
            df[col + "_hour"] = t.dt.hour.fillna(-1).astype(int)
            df[col + "_minute"] = t.dt.minute.fillna(-1).astype(int)
            df.drop(columns=[col], inplace=True)

        split_time("departure_time")
        split_time("arrival_time")

        for col in ["checkin_baggage", "cabin_baggage"]:
            df[col + "_num"] = (
                df[col].astype(str)
                .str.extract(r"(\d+)")
                .fillna(0)
                .astype(int)
            )
            df.drop(columns=[col], inplace=True)

        if "seats_available" not in df.columns:
            df["seats_available"] = seat_median

        for col, fmap in freq_maps.items():
            if col in df.columns:
                df[col + "_freq"] = df[col].map(fmap).fillna(0)

        X = df[features]
        predicted_price = float(bst.predict(xgb.DMatrix(X))[0])
        model_used = "ML Model (Raw)"

    
    else:
        predicted_price = fallback_prediction(current_price, days_left)
        model_used = "Rule-Based Engine"


    st.markdown("---")

    colA, colB = st.columns(2)
    colA.metric("Current Fare", f"₹{current_price}")
    colB.metric("Predicted Fare", f"₹{round(predicted_price,2)}")

    if model_used.startswith("ML"):
        st.success("Prediction powered by trained ML model.")
    else:
        st.warning("Route not in ML dataset. Using rule-based estimate.")
