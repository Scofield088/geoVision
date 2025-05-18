import ee
import geemap
import google.generativeai as genai
from django.shortcuts import render

# Initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# Configure Gemini API
genai.configure(api_key="AIzaSyD9yuxRq4K3fI35BARLgwqSkVbpwARwotw")

# Function to fetch and save satellite imagery
def get_satellite_image(latitude, longitude, year="2023"):
    region = ee.Geometry.Point([longitude, latitude]).buffer(5000).bounds()

    image = ee.ImageCollection("COPERNICUS/S2_SR") \
        .filterBounds(region) \
        .filterDate(f"{year}-01-01", f"{year}-12-31") \
        .sort("CLOUDY_PIXEL_PERCENTAGE") \
        .first() \
        .select(["B4", "B3", "B2"])  # RGB bands

    img_filename = f"static/Satellite_Image_{latitude}_{longitude}.png"
    geemap.ee_export_image(image, filename=img_filename, scale=10, region=region)

    return img_filename

# Function to get Carbon Emissions
def get_carbon_emissions(latitude, longitude, year="2023"):
    region = ee.Geometry.Point([longitude, latitude]).buffer(5000).bounds()

    dataset = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_CO') \
        .select('CO_column_number_density') \
        .filterDate(f'{year}-01-01', f'{year}-12-31') \
        .filterBounds(region)

    mean_co = dataset.mean()
    emission_value = mean_co.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=1000
    ).getInfo()

    if emission_value and 'CO_column_number_density' in emission_value:
        return round(emission_value['CO_column_number_density'], 6)
    return None

# Function to get AI-powered suggestions using Gemini API
def get_ai_suggestions(carbon_emission):
    if carbon_emission is None:
        return "No data available for this location."

    prompt = f"""
    The detected carbon emission level is {carbon_emission} mol/m². 
    Based on this level, provide recommendations to reduce carbon emissions.
    Include sustainable actions, policies, and technological solutions.
    Give everything in 15 lines.
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    return response.text if response.text else "Unable to generate suggestions at the moment."

# Django view to handle requests
def carbon_analysis(request):
    if request.method == "POST":
        latitude = float(request.POST.get("latitude"))
        longitude = float(request.POST.get("longitude"))

        # Fetch data
        satellite_image = get_satellite_image(latitude, longitude)
        carbon_emission = get_carbon_emissions(latitude, longitude)
        ai_suggestions = get_ai_suggestions(carbon_emission)

        return render(request, "carbon.html", {
            "satellite_image": satellite_image,
            "carbon_emission": carbon_emission,
            "ai_suggestions": ai_suggestions,
            "latitude": latitude,
            "longitude": longitude,
        })

    return render(request, "carbon.html")
