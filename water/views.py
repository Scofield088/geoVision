import ee
import geemap
from django.shortcuts import render

# Initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# Function to get water resources using Earth Engine
def get_water_resources(latitude, longitude, year="2023"):
    region = ee.Geometry.Point([longitude, latitude]).buffer(5000).bounds()

    # Using JRC Global Surface Water Dataset
    water_dataset = ee.Image("JRC/GSW1_4/GlobalSurfaceWater") \
        .select("occurrence")  # Water occurrence percentage

    water_masked = water_dataset.updateMask(water_dataset.gt(0))

    # Export the image
    water_image_path = f"static/Water_Resources_{latitude}_{longitude}.png"
    geemap.ee_export_image(water_masked, filename=water_image_path, scale=30, region=region)

    # Calculate mean water occurrence percentage
    water_info = water_masked.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=100
    ).getInfo()

    water_percentage = water_info.get("occurrence", "No Data")

    return water_image_path, water_percentage

# Django view for water resource management
def water_analysis(request):
    if request.method == "POST":
        latitude = float(request.POST.get("latitude"))
        longitude = float(request.POST.get("longitude"))

        # Fetch water resource data
        water_image, water_percentage = get_water_resources(latitude, longitude)

        return render(request, "water.html", {
            "water_image": water_image,
            "water_percentage": water_percentage,
            "latitude": latitude,
            "longitude": longitude,
        })

    return render(request, "water.html")
