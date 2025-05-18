import ee
import google.generativeai as genai
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Set up Gemini API Key
api_key = "AIzaSyD9yuxRq4K3fI35BARLgwqSkVbpwARwotw"  # 🔹 Replace with your actual API key
genai.configure(api_key=api_key)

# Initialize Google Earth Engine
try:
    ee.Initialize()
    print("Google Earth Engine Initialized!")
except Exception as e:
    print("Error initializing GEE:", e)

# Function to get AI suggestions
def get_gemini_suggestions(urban_area, vegetation_area):
    prompt = f"""
    The city has {urban_area} square meters of urbanized area and {vegetation_area} square meters of green vegetation.
    Suggest how to balance urban development while preserving nature.
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text  # Get text response
    except Exception as e:
        return f"Error: {str(e)}"

# Function to fetch satellite image data
def fetch_satellite_data(lat, lon):
    aoi = ee.Geometry.Point([lon, lat]).buffer(5000)  # 5km buffer

    image = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
        .filterBounds(aoi) \
        .filterDate("2023-01-01", "2023-12-31") \
        .median()

    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")

    urban_mask = ndbi.gt(0.2)
    vegetation_mask = ndvi.gt(0.3)

    urban_area = urban_mask.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e9
    ).getInfo().get("NDBI", 0)

    vegetation_area = vegetation_mask.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=30, maxPixels=1e9
    ).getInfo().get("NDVI", 0)

    # Generate GEE satellite image URL
    vis_params = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000}
    map_url = image.visualize(**vis_params).getThumbURL({
        "region": aoi, "scale": 30, "format": "png"
    })

    suggestions = get_gemini_suggestions(urban_area, vegetation_area)

    return {
        "urban_area": urban_area,
        "vegetation_area": vegetation_area,
        "satellite_image_url": map_url,
        "suggestions": suggestions
    }

# View for handling requests
def urban_analysis(request):
    if request.method == "POST":
        lat = float(request.POST.get("latitude"))
        lon = float(request.POST.get("longitude"))

        result = fetch_satellite_data(lat, lon)
        return JsonResponse(result)  # Send JSON response

    return render(request, "urban.html")