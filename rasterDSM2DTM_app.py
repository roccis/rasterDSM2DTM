import streamlit as st
import numpy as np
import rasterio
from scipy.ndimage import grey_opening
import math
import tempfile
import os
from rasterio.warp import transform_bounds
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import base64
import io

def dsm_to_dtm_metric(input_path, output_path, search_radius_meters=10.0):
    """
    Derives a DTM from a DSM using a metric window size.
    
    Args:
        input_path: Path to DSM geotiff
        output_path: Path to save DTM geotiff
        search_radius_meters: The physical width of the largest object to remove
        
    Returns:
        tuple: (dtm_path, chm_path, metadata_dict)
    """
    with rasterio.open(input_path) as src:
        # Get pixel resolution (assuming square pixels in meters)
        res_x, res_y = src.res 
        dsm = src.read(1)
        affine = src.transform
        crs = src.crs
        bounds = src.bounds

        # Calculate window size in pixels (quanta)
        # We use ceil to ensure the window is 'at least' as large as the radius
        window_pixels = math.ceil(search_radius_meters / res_x)
        
        # Ensure window_pixels is odd for a centered kernel (optional but recommended)
        if window_pixels % 2 == 0:
            window_pixels += 1

        # Handle NoData effectively
        nodata = src.nodata if src.nodata is not None else -9999

        # Change nodata to np.nan
        dsm_temp = dsm.astype('float32')
        dsm_temp[dsm_temp == nodata] = np.nanmedian(dsm)

        # Apply Morphology
        dtm = grey_opening(dsm_temp, size=(window_pixels, window_pixels))
        # After filtering, restore NoData values
        dtm[(dsm == nodata) | np.isnan(dsm)] = nodata

        # Metadata for output
        out_meta = src.meta.copy()
        out_meta.update(dtype=dtm.dtype, nodata=nodata)

        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(dtm, 1)

        # Canopy Height Model (CHM)
        chm = dsm - dtm
        chm[(dsm == nodata) | (dtm == nodata)] = nodata

        chm_path = output_path.replace('.tif', '_chm.tif')
        with rasterio.open(
            chm_path, 'w', driver='GTiff',
            height=chm.shape[0], width=chm.shape[1],
            count=1, dtype=chm.dtype,
            crs=crs, transform=affine, nodata=nodata
        ) as dst:
            dst.write(chm, 1)

        metadata = {
            'resolution': res_x,
            'window_pixels': window_pixels,
            'window_meters': search_radius_meters,
            'bounds': bounds,
            'crs': str(crs),
            'shape': dsm.shape
        }

        return output_path, chm_path, metadata


def get_raster_bounds_latlon(raster_path):
    """Get the bounds of a raster in lat/lon coordinates"""
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        # If not in WGS84, reproject bounds
        if src.crs.to_string() != 'EPSG:4326':
            bounds = transform_bounds(src.crs, 'EPSG:4326', *bounds)
        return bounds


def get_raster_data(raster_path):
    """Extract raster data and georeferencing info for plotly"""
    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999
        
        # Replace nodata with NaN
        data[data == nodata] = np.nan
        
        # Get transform and bounds
        transform = src.transform
        bounds = src.bounds  # (left, bottom, right, top)
        original_crs = src.crs
        
        # Convert bounds to lat/lon if needed
        if original_crs.to_string() != 'EPSG:4326':
            bounds = transform_bounds(original_crs, 'EPSG:4326', *bounds)
        
        # Create lat/lon grids for the raster
        height, width = data.shape
        lons = np.linspace(bounds[0], bounds[2], width)
        lats = np.linspace(bounds[3], bounds[1], height)  # Note: top to bottom for image coordinates
        
        return data, lons, lats, bounds


def raster_to_png_data_uri(data):
    """Convert raster array to a PNG data URI for Mapbox image layer."""
    data = data.astype(np.float32)
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        raise ValueError("No valid data in raster")

    vmin = np.nanpercentile(data, 2)
    vmax = np.nanpercentile(data, 98)
    scaled = np.zeros_like(data, dtype=np.uint8)
    scaled[valid_mask] = np.clip((data[valid_mask] - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

    rgb = np.stack([scaled, scaled, scaled], axis=2)
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# Streamlit App
st.set_page_config(page_title="DSM to DTM Converter", layout="wide")

# Hardcoded Mapbox token
MAPBOX_TOKEN = st.secrets["mapbox"]["token"]

st.title("🗺️ Raster DSM to DTM Converter")
st.markdown("Upload a Digital Surface Model (DSM) raster to generate a Digital Terrain Model (DTM) and Canopy Height Model (CHM)")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Window size slider
    window_size = st.slider(
        "Window Size (meters)",
        min_value=1.0,
        max_value=30.0,
        value=10.0,
        step=0.5,
        help="The physical width of the largest object to remove (e.g., trees, buildings)"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool uses morphological opening to derive a DTM from a DSM by removing 
    above-ground features like vegetation and buildings.
    
    **Window Size**: Larger windows remove bigger objects but may over-smooth terrain.
    """)

# File uploader
uploaded_file = st.file_uploader(
    "Upload DSM GeoTIFF",
    type=['tif', 'tiff'],
    help="Upload a Digital Surface Model in GeoTIFF format"
)

if uploaded_file is not None:
    # Create temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_input:
        tmp_input.write(uploaded_file.getvalue())
        input_path = tmp_input.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_output:
        output_path = tmp_output.name
    
    # Process button
    if st.button("🚀 Process DSM", type="primary"):
        with st.spinner(f"Processing with {window_size}m window..."):
            try:
                # Process the DSM
                dtm_path, chm_path, metadata = dsm_to_dtm_metric(
                    input_path, 
                    output_path, 
                    search_radius_meters=window_size
                )
                
                st.success("✅ Processing complete!")
                
                # Display metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Resolution", f"{metadata['resolution']:.2f}m")
                with col2:
                    st.metric("Window (pixels)", metadata['window_pixels'])
                with col3:
                    st.metric("Window (meters)", f"{metadata['window_meters']:.1f}m")
                
                # Visualization section
                st.header("📊 Results Visualization")
                
                # Get raster data with proper lat/lon bounds
                dsm_data, lons, lats, bounds = get_raster_data(input_path)
                center_lon = (bounds[0] + bounds[2]) / 2
                center_lat = (bounds[1] + bounds[3]) / 2

                # Build raster image overlay
                image_uri = raster_to_png_data_uri(dsm_data)
                west, south, east, north = bounds
                image_layer = {
                    "sourcetype": "image",
                    "source": image_uri,
                    "coordinates": [
                        [west, north],
                        [east, north],
                        [east, south],
                        [west, south]
                    ],
                    "opacity": 0.7
                }

                # Initialize mapbox with plotly express
                fig = px.scatter_mapbox(
                    lat=[center_lat],
                    lon=[center_lon],
                    zoom=12,
                )
                fig.update_layout(
                    mapbox_style="satellite",
                    mapbox_accesstoken=MAPBOX_TOKEN,
                    mapbox_layers=[image_layer],
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=600,
                )

                st.plotly_chart(fig, use_container_width=True)
                
                # Download buttons
                st.header("💾 Download Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    with open(dtm_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download DTM",
                            data=f,
                            file_name="dtm.tif",
                            mime="image/tiff"
                        )
                
                with col2:
                    with open(chm_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download CHM",
                            data=f,
                            file_name="chm.tif",
                            mime="image/tiff"
                        )
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
            
            finally:
                # Cleanup temporary files
                if os.path.exists(input_path):
                    os.unlink(input_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
else:
    st.info("👆 Upload a DSM GeoTIFF file to get started")