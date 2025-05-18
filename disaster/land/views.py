from django.shortcuts import render
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage import filters, morphology
from scipy import ndimage
from datetime import datetime, timedelta
import ee
import folium
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import os
from IPython.display import display
from google.oauth2 import service_account
from geopy.geocoders import Nominatim

credentials_path = './credentials.json'
scopes = ['https://www.googleapis.com/auth/earthengine']

credentials = service_account.Credentials.from_service_account_file(credentials_path, scopes=scopes)

# Initialize Earth Engine
ee.Initialize(credentials)
project_name = 'ee-mgoutham1975'


def get_earth_engine_image(location, date_str):
    """
    Function to retrieve and process Earth Engine images based on location and date
    
    Args:
        location (str): Comma-separated lat,lng coordinates
        date_str (str): Date in YYYY-MM-DD format
    
    Returns:
        tuple: (current_image, previous_year_image, map_object)
    """
    # Parse location
    try:
        lat, lng = map(float, location.split(','))
        ee_point = ee.Geometry.Point([lng, lat])
    except ValueError:
        print("Invalid location format. Using default (San Francisco).")
        lat, lng = 37.7749, -122.4194
        ee_point = ee.Geometry.Point([-122.4194, 37.7749])
    
    # Parse date
    try:
        target_date = ee.Date(date_str)
    except:
        print(f"Invalid date format: {date_str}. Using current date.")
        target_date = ee.Date(datetime.now().strftime('%Y-%m-%d'))
    
    # Calculate previous year date
    previous_year_date = ee.Date(target_date).advance(-1, 'year')
    
    # Function to get satellite imagery for a given location and date
    def get_image_for_date(location, date):
        # Create a date range to search for images (5-day window to ensure coverage)
        date_range = ee.DateRange(ee.Date(date).advance(-2, 'day'), 
                                ee.Date(date).advance(2, 'day'))
        
        # Load Sentinel-2 surface reflectance collection
        sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR') \
            .filterBounds(location) \
            .filterDate(date_range) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        
        # If no Sentinel-2 images available, try Landsat 8
        if sentinel2.size().getInfo() == 0:
            landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                .filterBounds(location) \
                .filterDate(date_range)
            
            image = landsat.sort('CLOUD_COVER').first()
            
            # Convert Landsat 8 to RGB visualization
            if image:
                image = image.select(['SR_B4', 'SR_B3', 'SR_B2']) \
                        .divide(10000) \
                        .rename(['B4', 'B3', 'B2'])
        else:
            # Use Sentinel-2 imagery
            image = sentinel2.sort('CLOUDY_PIXEL_PERCENTAGE').first()
            if image:
                image = image.select(['B4', 'B3', 'B2']) \
                        .divide(10000)
        
        return image
    
    # Get images for both dates
    current_image = get_image_for_date(ee_point, target_date)
    previous_year_image = get_image_for_date(ee_point, previous_year_date)
    
    # Create visualization parameters
    vis_params = {
        'min': 0.0,
        'max': 0.3,
        'bands': ['B4', 'B3', 'B2']
    }
    
    # Create a map centered at the location
    map_object = folium.Map(location=[lat, lng], zoom_start=12)
    
    # Add the Earth Engine layers to the map
    if current_image:
        map_id = current_image.getMapId(vis_params)
        folium.TileLayer(
            tiles=map_id['tile_fetcher'].url_format,
            attr='Google Earth Engine',
            name=f'Current Date: {date_str}',
            overlay=True
        ).add_to(map_object)
    
    if previous_year_image:
        map_id = previous_year_image.getMapId(vis_params)
        folium.TileLayer(
            tiles=map_id['tile_fetcher'].url_format,
            attr='Google Earth Engine',
            name=f'Previous Year: {previous_year_date.format("YYYY-MM-dd").getInfo()}',
            overlay=True
        ).add_to(map_object)
    
    # Add layer control
    folium.LayerControl().add_to(map_object)
    
    return current_image, previous_year_image, map_object

def save_ee_image(image, filename, region, scale=30):
    """
    Download and save an Earth Engine image locally
    
    Args:
        image (ee.Image): Earth Engine image object
        filename (str): Output filename
        region (ee.Geometry): Region to download
        scale (int): Resolution in meters
    
    Returns:
        PIL.Image or None: The downloaded image if successful, None otherwise
    """
    if image is None:
        print(f"No image data available for {filename}")
        return None
    
    # Set up visualization parameters
    vis_params = {
        'min': 0.0,
        'max': 0.3,
        'bands': ['B4', 'B3', 'B2'],
        'format': 'png'
    }
    
    # Get the image URL
    url = image.getThumbURL({
        'region': region,
        'dimensions': '1024',
        'format': 'png',
        'min': 0.0,
        'max': 0.3,
        'bands': ['B4', 'B3', 'B2']
    })
    
    # Download the image
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img.save(filename)
        print(f"Image saved as {filename}")
        return img
    else:
        print(f"Failed to download image: {response.status_code}")
        return None

def display_images(current_img, previous_img, current_date, previous_date):
    """
    Display two images side by side with matplotlib
    
    Args:
        current_img: PIL Image of current date
        previous_img: PIL Image of previous date
        current_date: String representation of current date
        previous_date: String representation of previous date
    """
    if current_img and previous_img:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        ax1.imshow(current_img)
        ax1.set_title(f"Current Date: {current_date}")
        ax1.axis('off')
        
        ax2.imshow(previous_img)
        ax2.set_title(f"Previous Year: {previous_date}")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("Unable to display images - one or both images are missing.")

def create_grayscale_masks(current_img_path, previous_img_path):
    """
    Create grayscale masks from the RGB images
    
    Args:
        current_img_path: Path to current date image
        previous_img_path: Path to previous date image
        
    Returns:
        tuple: Paths to the created mask files
    """
    current_mask_path = "current_mask.png"
    previous_mask_path = "previous_mask.png"
    
    # Process current image
    try:
        img = Image.open(current_img_path)
        gray_img = img.convert("L")
        gray_img.save(current_mask_path)
        print(f"Created grayscale mask: {current_mask_path}")
    except Exception as e:
        print(f"Error creating current mask: {e}")
        current_mask_path = None
        
    # Process previous image
    try:
        img = Image.open(previous_img_path)
        gray_img = img.convert("L")
        gray_img.save(previous_mask_path)
        print(f"Created grayscale mask: {previous_mask_path}")
    except Exception as e:
        print(f"Error creating previous mask: {e}")
        previous_mask_path = None
        
    return current_mask_path, previous_mask_path

# Function to extract landform features from an image
def extract_landform_features(image, mask):
    """
    Extract landform features from the image using the mask as a guide

    Parameters:
    image: Tensor - Image to extract features from
    mask: Tensor - Mask indicating areas of interest

    Returns:
    dict: Extracted landform features
    """
    # Convert tensors to numpy for processing
    img_np = image[0].permute(1, 2, 0).cpu().numpy()
    mask_np = mask[0, 0].cpu().numpy() > 0.5  # Convert to binary mask

    # Extract grayscale image for elevation analysis
    gray_img = np.mean(img_np, axis=2)

    # Apply the mask to focus on relevant areas
    masked_gray = gray_img * mask_np

    # Edge detection to identify terrain boundaries
    edges = filters.sobel(masked_gray)

    # Identify potential water bodies (usually darker regions)
    potential_water = (masked_gray < 0.3) & mask_np
    water_regions = morphology.remove_small_objects(potential_water, min_size=50)

    # Identify potential elevated areas (usually brighter regions)
    potential_elevated = (masked_gray > 0.7) & mask_np
    elevated_regions = morphology.remove_small_objects(potential_elevated, min_size=50)

    # Identify potential vegetation (green channel prominence)
    vegetation_index = (img_np[:,:,1] - np.mean(img_np[:,:,[0,2]], axis=2)) * mask_np
    vegetation_areas = (vegetation_index > 0.1) & mask_np
    vegetation_regions = morphology.remove_small_objects(vegetation_areas, min_size=50)

    # Calculate terrain roughness (local variation)
    smoothed = ndimage.gaussian_filter(masked_gray, sigma=2)
    roughness = np.abs(masked_gray - smoothed) * mask_np
    rough_terrain = roughness > np.percentile(roughness[mask_np], 75)

    # Calculate area statistics
    total_area = np.sum(mask_np)
    water_area = np.sum(water_regions)
    elevated_area = np.sum(elevated_regions)
    vegetation_area = np.sum(vegetation_regions)
    rough_area = np.sum(rough_terrain)

    features = {
        "total_area": total_area,
        "water_area": water_area,
        "water_percentage": (water_area / total_area * 100) if total_area > 0 else 0,
        "elevated_area": elevated_area,
        "elevated_percentage": (elevated_area / total_area * 100) if total_area > 0 else 0,
        "vegetation_area": vegetation_area,
        "vegetation_percentage": (vegetation_area / total_area * 100) if total_area > 0 else 0,
        "rough_terrain_area": rough_area,
        "rough_terrain_percentage": (rough_area / total_area * 100) if total_area > 0 else 0,
        "mean_elevation": np.mean(masked_gray[mask_np]) if np.any(mask_np) else 0,
        "edge_density": np.sum(edges) / total_area if total_area > 0 else 0,
        "visualization": {
            "original_image": img_np,
            "mask": mask_np,
            "elevation_map": masked_gray,
            "edges": edges,
            "water_regions": water_regions,
            "elevated_regions": elevated_regions,
            "vegetation_regions": vegetation_regions,
            "rough_terrain": rough_terrain
        }
    }

    return features

# Function to assess landform changes
def assess_landform_changes(pre_features, post_features):
    """
    Assess changes in landforms between pre and post-disaster

    Parameters:
    pre_features: dict - Landform features before disaster
    post_features: dict - Landform features after disaster

    Returns:
    dict: Assessment results with change statistics
    """
    # Calculate changes in different features
    water_area_change = post_features["water_area"] - pre_features["water_area"]
    water_percentage_change = post_features["water_percentage"] - pre_features["water_percentage"]

    elevated_area_change = post_features["elevated_area"] - pre_features["elevated_area"]
    elevated_percentage_change = post_features["elevated_percentage"] - pre_features["elevated_percentage"]

    vegetation_area_change = post_features["vegetation_area"] - pre_features["vegetation_area"]
    vegetation_percentage_change = post_features["vegetation_percentage"] - pre_features["vegetation_percentage"]

    rough_terrain_change = post_features["rough_terrain_area"] - pre_features["rough_terrain_area"]
    rough_percentage_change = post_features["rough_terrain_percentage"] - pre_features["rough_terrain_percentage"]

    edge_density_change = post_features["edge_density"] - pre_features["edge_density"]
    mean_elevation_change = post_features["mean_elevation"] - pre_features["mean_elevation"]

    # Calculate overall terrain change intensity
    change_factors = [
        abs(water_percentage_change / 100) if pre_features["water_percentage"] > 0 else
            (post_features["water_percentage"] / 100 if post_features["water_percentage"] > 0 else 0),
        abs(elevated_percentage_change / 100) if pre_features["elevated_percentage"] > 0 else
            (post_features["elevated_percentage"] / 100 if post_features["elevated_percentage"] > 0 else 0),
        abs(vegetation_percentage_change / 100) if pre_features["vegetation_percentage"] > 0 else
            (post_features["vegetation_percentage"] / 100 if post_features["vegetation_percentage"] > 0 else 0),
        abs(rough_percentage_change / 100) if pre_features["rough_terrain_percentage"] > 0 else
            (post_features["rough_terrain_percentage"] / 100 if post_features["rough_terrain_percentage"] > 0 else 0),
        abs(edge_density_change / pre_features["edge_density"]) if pre_features["edge_density"] > 0 else 0
    ]

    overall_change_intensity = np.mean([cf for cf in change_factors if not np.isnan(cf)]) * 100

    # Interpret the changes
    if overall_change_intensity < 10:
        change_severity = "Minimal landform changes"
    elif overall_change_intensity < 25:
        change_severity = "Moderate landform changes"
    elif overall_change_intensity < 50:
        change_severity = "Significant landform changes"
    else:
        change_severity = "Extreme landform transformation"

    # Identify primary change types
    change_types = []

    if water_percentage_change > 5:
        change_types.append("Increased water coverage (possible flooding)")
    elif water_percentage_change < -5:
        change_types.append("Decreased water coverage (possible drought or drainage)")

    if elevated_percentage_change > 5:
        change_types.append("Increased elevated areas (possible debris accumulation)")
    elif elevated_percentage_change < -5:
        change_types.append("Decreased elevated areas (possible erosion or subsidence)")

    if vegetation_percentage_change > 5:
        change_types.append("Increased vegetation coverage")
    elif vegetation_percentage_change < -5:
        change_types.append("Decreased vegetation coverage (possible deforestation or burning)")

    if rough_percentage_change > 5:
        change_types.append("Increased terrain roughness (possible destabilization)")
    elif rough_percentage_change < -5:
        change_types.append("Decreased terrain roughness (possible smoothing or settling)")

    if edge_density_change > 0.05:
        change_types.append("Increased terrain edges (possible fracturing)")
    elif edge_density_change < -0.05:
        change_types.append("Decreased terrain edges (possible smoothing)")

    # If no specific changes are identified but overall change is detected
    if not change_types and overall_change_intensity > 10:
        change_types.append("General terrain alteration without specific pattern")

    assessment = {
        "water_area_change": water_area_change,
        "water_percentage_change": water_percentage_change,
        "elevated_area_change": elevated_area_change,
        "elevated_percentage_change": elevated_percentage_change,
        "vegetation_area_change": vegetation_area_change,
        "vegetation_percentage_change": vegetation_percentage_change,
        "rough_terrain_change": rough_terrain_change,
        "rough_percentage_change": rough_percentage_change,
        "edge_density_change": edge_density_change,
        "mean_elevation_change": mean_elevation_change,
        "overall_change_intensity": overall_change_intensity,
        "change_severity": change_severity,
        "change_types": change_types,
        "pre_features": pre_features,
        "post_features": post_features
    }

    return assessment

# Visualize results
def visualize_landform_changes(assessment):
    """
    Create visualization of landform changes

    Parameters:
    assessment: dict - Assessment results from assess_landform_changes function
    
    Returns:
    matplotlib.figure.Figure: The visualization figure
    """
    # Create figure with subplots
    fig, axs = plt.subplots(3, 4, figsize=(20, 15))

    # Row 1: Original images and masks
    axs[0, 0].imshow(assessment["pre_features"]["visualization"]["original_image"])
    axs[0, 0].set_title("Pre-disaster Image")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(assessment["pre_features"]["visualization"]["mask"], cmap="gray")
    axs[0, 1].set_title("Pre-disaster Mask")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(assessment["post_features"]["visualization"]["original_image"])
    axs[0, 2].set_title("Post-disaster Image")
    axs[0, 2].axis("off")

    axs[0, 3].imshow(assessment["post_features"]["visualization"]["mask"], cmap="gray")
    axs[0, 3].set_title("Post-disaster Mask")
    axs[0, 3].axis("off")

    # Row 2: Elevation maps and edge detection
    axs[1, 0].imshow(assessment["pre_features"]["visualization"]["elevation_map"], cmap="terrain")
    axs[1, 0].set_title("Pre-disaster Elevation Map")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(assessment["pre_features"]["visualization"]["edges"], cmap="gray")
    axs[1, 1].set_title("Pre-disaster Edge Detection")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(assessment["post_features"]["visualization"]["elevation_map"], cmap="terrain")
    axs[1, 2].set_title("Post-disaster Elevation Map")
    axs[1, 2].axis("off")

    axs[1, 3].imshow(assessment["post_features"]["visualization"]["edges"], cmap="gray")
    axs[1, 3].set_title("Post-disaster Edge Detection")
    axs[1, 3].axis("off")

    # Row 3: Feature comparisons and summary
    # Combine water and vegetation features
    pre_features_combined = np.zeros((*assessment["pre_features"]["visualization"]["mask"].shape, 3))
    pre_features_combined[:,:,0] = assessment["pre_features"]["visualization"]["elevated_regions"]  # Red for elevation
    pre_features_combined[:,:,1] = assessment["pre_features"]["visualization"]["vegetation_regions"]  # Green for vegetation
    pre_features_combined[:,:,2] = assessment["pre_features"]["visualization"]["water_regions"]  # Blue for water

    post_features_combined = np.zeros((*assessment["post_features"]["visualization"]["mask"].shape, 3))
    post_features_combined[:,:,0] = assessment["post_features"]["visualization"]["elevated_regions"]  # Red for elevation
    post_features_combined[:,:,1] = assessment["post_features"]["visualization"]["vegetation_regions"]  # Green for vegetation
    post_features_combined[:,:,2] = assessment["post_features"]["visualization"]["water_regions"]  # Blue for water

    axs[2, 0].imshow(pre_features_combined)
    axs[2, 0].set_title("Pre-disaster Features\nRed=Elevation, Green=Vegetation, Blue=Water")
    axs[2, 0].axis("off")

    axs[2, 1].imshow(post_features_combined)
    axs[2, 1].set_title("Post-disaster Features\nRed=Elevation, Green=Vegetation, Blue=Water")
    axs[2, 1].axis("off")

    # Create change visualization
    change_visualization = np.abs(
        assessment["post_features"]["visualization"]["elevation_map"] -
        assessment["pre_features"]["visualization"]["elevation_map"]
    )

    # Normalize for better visualization
    if np.max(change_visualization) > 0:
        change_visualization = change_visualization / np.max(change_visualization)

    axs[2, 2].imshow(change_visualization, cmap="hot")
    axs[2, 2].set_title("Elevation Change Intensity")
    axs[2, 2].axis("off")

    # Summary text
    axs[2, 3].axis("off")
    change_types_text = "\n".join(assessment["change_types"]) if assessment["change_types"] else "No significant specific changes detected"

    summary_text = f"LANDFORM CHANGE ASSESSMENT\n\n" \
                   f"Change Severity: {assessment['change_severity']}\n" \
                   f"Overall Change Intensity: {assessment['overall_change_intensity']:.2f}%\n\n" \
                   f"Primary Change Types:\n{change_types_text}\n\n" \
                   f"Water Coverage: {assessment['water_percentage_change']:.2f}%\n" \
                   f"Elevated Areas: {assessment['elevated_percentage_change']:.2f}%\n" \
                   f"Vegetation: {assessment['vegetation_percentage_change']:.2f}%\n" \
                   f"Terrain Roughness: {assessment['rough_percentage_change']:.2f}%\n" \
                   f"Elevation Change: {assessment['mean_elevation_change']:.2f}"

    axs[2, 3].text(0, 0, summary_text, fontsize=12, va="top")

    plt.tight_layout()
    plt.savefig("landform_change_assessment.png", dpi=300)
    plt.show()

    return fig
        
def load_image_and_mask(img_path, mask_path, transform):
    """
    Load and preprocess images
    
    Args:
        img_path: Path to the image file
        mask_path: Path to the mask file
        transform: Transformations to apply
        
    Returns:
        tuple: (transformed image, transformed mask)
    """
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")  # Grayscale mask

    if transform:
        img = transform(img)
        mask = transform(mask)

    return img, mask




def land_analysis_view(request):
    if request.method == 'POST':
        place_name = request.POST.get('location', '')
        geolocator = Nominatim(user_agent="my_geocoder")
        x = geolocator.geocode(place_name)
        
        if not x:
            return render(request, 'land.html', {'error': 'Location not found'})
        
        location = f"{x.latitude},{x.longitude}"
        
        today = datetime.now()
        two_days_ago = today - timedelta(days=2)
        one_year_ago = today - timedelta(days=365)

        date_str_2_days = two_days_ago.strftime("%Y-%m-%d")

        current_image, previous_year_image, map_object = get_earth_engine_image(location, date_str_2_days)
        output_map_file = "static/earth_engine_comparison.html"
        map_object.save(output_map_file)

        lat, lng = map(float, location.split(','))
        region = ee.Geometry.Point([lng, lat]).buffer(5000)

        current_img_file = "static/current.png"
        previous_img_file = "static/previous.png"

        current_img = save_ee_image(current_image, current_img_file, region)
        previous_img = save_ee_image(previous_year_image, previous_img_file, region)

        if current_img and previous_img:
            current_mask_path, previous_mask_path = create_grayscale_masks(current_img_file, previous_img_file)
            
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
            
            pre_img, pre_mask = load_image_and_mask(previous_img_file, previous_mask_path, transform)
            post_img, post_mask = load_image_and_mask(current_img_file, current_mask_path, transform)
            
            pre_img = pre_img.unsqueeze(0)
            pre_mask = pre_mask.unsqueeze(0)
            post_img = post_img.unsqueeze(0)
            post_mask = post_mask.unsqueeze(0)
            
            pre_features = extract_landform_features(pre_img, pre_mask)
            post_features = extract_landform_features(post_img, post_mask)
            
            assessment = assess_landform_changes(pre_features, post_features)
            vis_figure = visualize_landform_changes(assessment)
            
            return render(request, 'land.html', {
                'map_file': output_map_file,
                'current_image': current_img_file,
                'previous_image': previous_img_file,
                'assessment': assessment,
            })
    
    return render(request, 'land.html')