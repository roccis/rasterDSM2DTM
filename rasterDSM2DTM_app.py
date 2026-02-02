import streamlit as st
import numpy as np
import rasterio
from scipy.ndimage import grey_opening
import math
import tempfile
import os
import pydeck as pdk
from rasterio.warp import calculate_default_transform, reproject, Resampling
from PIL import Image
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
            from rasterio.warp import transform_bounds
            bounds = transform_bounds(src.crs, 'EPSG:4326', *bounds)
        return bounds


def create_raster_overlay(raster_path, colormap='terrain'):
    """Create a bitmap overlay for pydeck"""
    with rasterio.open(raster_path) as src:
        # Read the data
        data = src.read(1)
        nodata = src.nodata if src.nodata is not None else -9999
        
        # Mask nodata
        data_masked = np.ma.masked_equal(data, nodata)
        
        # Normalize to 0-255
        vmin, vmax = np.percentile(data_masked.compressed(), [2, 98])
        normalized = np.clip((data_masked - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        
        # Convert to RGB using a colormap
        if colormap == 'terrain':
            # Brown to green gradient
            r = np.clip(255 - normalized * 0.5, 100, 255).astype(np.uint8)
            g = np.clip(normalized * 0.8, 100, 200).astype(np.uint8)
            b = np.clip(normalized * 0.3, 50, 150).astype(np.uint8)
        elif colormap == 'height':
            # Blue to red for height
            r = normalized
            g = 255 - normalized
            b = 128
        else:
            r = g = b = normalized
        
        # Create RGBA image
        rgba = np.dstack([r, g, b, np.where(data == nodata, 0, 180)])
        
        # Get bounds in lat/lon
        bounds = get_raster_bounds_latlon(raster_path)
        
        return rgba, bounds


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
                
                # Get bounds for initial view
                bounds = get_raster_bounds_latlon(input_path)
                center_lon = (bounds[0] + bounds[2]) / 2
                center_lat = (bounds[1] + bounds[3]) / 2
                    
                # Create overlays
                dsm_overlay, dsm_bounds = create_raster_overlay(input_path, 'terrain')
                dtm_overlay, dtm_bounds = create_raster_overlay(dtm_path, 'terrain')
                chm_overlay, chm_bounds = create_raster_overlay(chm_path, 'height')
                
                # Tabs for different views
                tab1, tab2, tab3 = st.tabs(["📍 DSM", "🏔️ DTM", "🌳 CHM"])
                
                with tab1:
                    st.subheader("Digital Surface Model (Original)")
                    
                    # Save overlay to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
                        Image.fromarray(dsm_overlay).save(tmp_img.name)
                        
                        layer = pdk.Layer(
                            "BitmapLayer",
                            image=tmp_img.name,
                            bounds=[[dsm_bounds[0], dsm_bounds[1]], [dsm_bounds[2], dsm_bounds[3]]],
                            opacity=0.7
                        )
                        
                        view_state = pdk.ViewState(
                            longitude=center_lon,
                            latitude=center_lat,
                            zoom=15,
                            pitch=0
                        )
                        
                        r = pdk.Deck(
                            layers=[layer],
                            initial_view_state=view_state,
                            map_style="mapbox://styles/mapbox/satellite-v9",
                            mapbox_key=MAPBOX_TOKEN
                        )
                        
                        st.pydeck_chart(r, use_container_width=True)
                
                with tab2:
                    st.subheader("Digital Terrain Model (Ground)")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
                        Image.fromarray(dtm_overlay).save(tmp_img.name)
                        
                        layer = pdk.Layer(
                            "BitmapLayer",
                            image=tmp_img.name,
                            bounds=[[dtm_bounds[0], dtm_bounds[1]], [dtm_bounds[2], dtm_bounds[3]]],
                            opacity=0.7
                        )
                        
                        view_state = pdk.ViewState(
                            longitude=center_lon,
                            latitude=center_lat,
                            zoom=15,
                            pitch=0
                        )
                        
                        r = pdk.Deck(
                            layers=[layer],
                            initial_view_state=view_state,
                            map_style="mapbox://styles/mapbox/satellite-v9",
                            mapbox_key=MAPBOX_TOKEN
                        )
                        
                        st.pydeck_chart(r, use_container_width=True)
                
                with tab3:
                    st.subheader("Canopy Height Model (DSM - DTM)")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img:
                        Image.fromarray(chm_overlay).save(tmp_img.name)
                        
                        layer = pdk.Layer(
                            "BitmapLayer",
                            image=tmp_img.name,
                            bounds=[[chm_bounds[0], chm_bounds[1]], [chm_bounds[2], chm_bounds[3]]],
                            opacity=0.7
                        )
                        
                        view_state = pdk.ViewState(
                            longitude=center_lon,
                            latitude=center_lat,
                            zoom=15,
                            pitch=0
                        )
                        
                        r = pdk.Deck(
                            layers=[layer],
                            initial_view_state=view_state,
                            map_style="mapbox://styles/mapbox/satellite-v9",
                            mapbox_key=MAPBOX_TOKEN
                        )
                        
                        st.pydeck_chart(r, use_container_width=True)
                
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
            
            finally:
                # Cleanup temporary files
                if os.path.exists(input_path):
                    os.unlink(input_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
else:
    st.info("👆 Upload a DSM GeoTIFF file to get started")
    
    # Show example
    with st.expander("📖 Usage Instructions"):
        st.markdown("""
        1. **Upload your DSM**: Click the upload button and select your GeoTIFF file
        2. **Configure window size**: Use the slider in the sidebar (1-30 meters)
           - Smaller windows (1-5m): Remove small vegetation
           - Medium windows (5-15m): Remove trees and small buildings
           - Large windows (15-30m): Remove large buildings and tree canopies
        3. **Process**: Click the process button to generate DTM and CHM
        4. **View results**: Explore the interactive maps in different tabs
        5. **Download**: Save the generated DTM and CHM files
        """)