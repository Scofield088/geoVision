import ee
import google.generativeai as genai
from django.shortcuts import render
from django.http import JsonResponse

# Initialize Google Earth Engine
ee.Initialize()

# Set up Gemini API
genai.configure(api_key="AIzaSyD9yuxRq4K3fI35BARLgwqSkVbpwARwotw")

# Function to analyze forest data
def analyze_forest():
    # Define Area of Interest (Modify Coordinates)
    aoi = ee.Geometry.Polygon([
        [[-60.0, -10.0], [-60.0, -12.0], [-58.0, -12.0], [-58.0, -10.0]]
    ])

    # Load Landsat 8 Collection and Compute NDVI
    landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA') \
        .filterBounds(aoi) \
        .filterDate('2020-01-01', '2024-01-01') \
        .median()

    ndvi = landsat.normalizedDifference(['B5', 'B4']).rename('NDVI')

    # Load Updated MODIS Land Cover Dataset
    modis_landcover = ee.ImageCollection('MODIS/061/MCD12Q1') \
        .filterDate('2020-01-01', '2024-01-01') \
        .select('LC_Type1') \
        .mode().clip(aoi)

    # Load Updated Hansen Global Forest Change Dataset
    forest_loss = ee.Image('UMD/hansen/global_forest_change_2023_v1_11') \
        .select('lossyear') \
        .clip(aoi)

    # Calculate Total Forested Area in 2000
    forest_2000 = ee.Image('UMD/hansen/global_forest_change_2023_v1_11') \
        .select('treecover2000') \
        .gt(30).clip(aoi)

    # Compute Deforestation Rate
    loss_area = forest_loss.gt(0).multiply(ee.Image.pixelArea())  # Lost forest area
    forest_area_2000 = forest_2000.multiply(ee.Image.pixelArea())  # Initial forest area

    total_loss = loss_area.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e13
    )
    total_forest_2000 = forest_area_2000.reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e13
    )

    deforestation_rate = ee.Number(total_loss.get('lossyear')) \
        .divide(ee.Number(total_forest_2000.get('treecover2000'))) \
        .multiply(100).getInfo()

    # Compute Vegetative Land Coverage
    vegetative_land = ndvi.gt(0.3)
    veg_area = vegetative_land.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e13
    )

    vegetative_land_area = veg_area.getInfo()['NDVI'] / 1e6  # Convert to sq. km

    return deforestation_rate, vegetative_land_area

# Function to Get AI-Powered Suggestions from Gemini API
def get_ai_suggestions(deforestation_rate, vegetative_land_area):
    prompt = f"""
    The deforestation rate is {deforestation_rate:.2f}% and vegetative land area is {vegetative_land_area:.2f} sq. km.
    Provide actionable strategies to reduce deforestation and improve vegetation.
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text if response else "No response from AI."

# Django View
def veg_analysis(request):
    deforestation_rate, vegetative_land_area = analyze_forest()
    ai_suggestions = get_ai_suggestions(deforestation_rate, vegetative_land_area)

    context = {
        'deforestation_rate': f"{deforestation_rate:.2f}%",
        'vegetative_land_area': f"{vegetative_land_area:.2f} sq. km",
        'ai_suggestions': ai_suggestions
    }

    return render(request, 'veg.html', context)
