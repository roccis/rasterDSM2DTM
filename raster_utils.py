"""
Utility functions for raster processing and visualization.
"""

import numpy as np
import rasterio
from scipy.ndimage import grey_opening
import math
from rasterio.warp import transform_bounds
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import base64
import io


def dsm_to_dtm_metric(input_path, output_path, search_radius_meters=10.0, max_pixels=15000000):
    """
    Derives a DTM from a DSM using a metric window size.
    
    Args:
        input_path: Path to DSM geotiff
        output_path: Path to save DTM geotiff
        search_radius_meters: The physical width of the largest object to remove
        max_pixels: Maximum number of pixels to process (default 15M = ~3873x3873)
        
    Returns:
        tuple: (dtm_path, chm_path, metadata_dict)
    """
    with rasterio.open(input_path) as src:
        # Get pixel resolution (assuming square pixels in meters)
        res_x, res_y = src.res 
        height, width = src.height, src.width
        total_pixels = height * width
        
        # Always read original DSM at full resolution for CHM calculation
        dsm_original = src.read(1)
        original_nodata = src.nodata if src.nodata is not None else -9999
        
        # Calculate window size in pixels BEFORE downsampling (based on original resolution)
        # This ensures morphological filtering is consistent regardless of downsampling
        window_pixels = math.ceil(search_radius_meters / res_x)
        
        # Ensure window_pixels is odd for a centered kernel
        if window_pixels % 2 == 0:
            window_pixels += 1
        
        # Calculate downsampling if needed to stay within memory limits
        downsample = 1
        if total_pixels > max_pixels:
            downsample = math.ceil(math.sqrt(total_pixels / max_pixels))
            out_height = height // downsample
            out_width = width // downsample
            # Read with downsampling
            dsm = src.read(
                1,
                out_shape=(out_height, out_width),
                resampling=rasterio.enums.Resampling.average
            )
            # Update transform for downsampled raster
            from rasterio.transform import Affine
            affine = src.transform * Affine.scale(downsample, downsample)
        else:
            dsm = src.read(1)
            affine = src.transform
            
        crs = src.crs
        bounds = src.bounds
        nodata = src.nodata if src.nodata is not None else -9999

        # Handle NoData effectively
        dsm_temp = dsm.astype('float32')
        dsm_temp[dsm_temp == nodata] = np.nanmedian(dsm)

        # Apply Morphology with window size from original resolution
        # This keeps the filtering consistent even if data is downsampled
        dtm = grey_opening(dsm_temp, size=(window_pixels, window_pixels))
        # After filtering, restore NoData values
        dtm[(dsm == nodata) | np.isnan(dsm)] = nodata

        # Metadata for output
        out_meta = src.meta.copy()
        out_meta.update(
            dtype=dtm.dtype,
            nodata=nodata,
            height=dtm.shape[0],
            width=dtm.shape[1],
            transform=affine
        )

        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(dtm, 1)

        # Canopy Height Model (CHM) - computed at original DSM resolution
        # If DTM was downsampled, upsample it back to original DSM resolution
        if downsample > 1:
            # Resample DTM back to original resolution
            from rasterio.warp import reproject, Resampling
            dtm_original = np.zeros_like(dsm_original, dtype=dtm.dtype)
            reproject(
                dtm,
                dtm_original,
                src_transform=affine,
                dst_transform=src.transform,
                src_crs=crs,
                dst_crs=crs,
                resampling=Resampling.bilinear
            )
        else:
            dtm_original = dtm
        
        chm = dsm_original - dtm_original
        chm[(dsm_original == original_nodata) | (dtm_original == nodata)] = original_nodata

        chm_path = output_path.replace('.tif', '_chm.tif')
        original_meta = src.meta.copy()
        original_meta.update(
            dtype=chm.dtype,
            nodata=original_nodata
        )
        with rasterio.open(chm_path, 'w', **original_meta) as dst:
            dst.write(chm, 1)

        metadata = {
            'resolution': res_x,
            'window_pixels': window_pixels,
            'window_meters': search_radius_meters,
            'bounds': bounds,
            'crs': str(crs),
            'shape': dsm.shape,
            'original_shape': (height, width),
            'downsampled': downsample > 1,
            'downsample_factor': downsample
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


def get_raster_data(raster_path, max_dim=2000):
    """Extract raster data and georeferencing info for plotly.
    
    Args:
        raster_path: Path to raster file
        max_dim: Maximum dimension for output (will downsample if needed)
    """
    with rasterio.open(raster_path) as src:
        # Calculate downsampling factor if needed
        height, width = src.height, src.width
        downsample = max(1, math.ceil(max(height, width) / max_dim))
        
        if downsample > 1:
            # Read with downsampling
            data = src.read(
                1,
                out_shape=(height // downsample, width // downsample),
                resampling=rasterio.enums.Resampling.average
            ).astype(np.float32)
        else:
            data = src.read(1).astype(np.float32)
        
        nodata = src.nodata if src.nodata is not None else -9999
        
        # Replace nodata with NaN
        data[data == nodata] = np.nan
        
        # Get bounds
        bounds = src.bounds  # (left, bottom, right, top)
        original_crs = src.crs
        
        # Convert bounds to lat/lon if needed
        if original_crs.to_string() != 'EPSG:4326':
            bounds = transform_bounds(original_crs, 'EPSG:4326', *bounds)
        
        # Create lat/lon grids for the raster
        out_height, out_width = data.shape
        lons = np.linspace(bounds[0], bounds[2], out_width)
        lats = np.linspace(bounds[3], bounds[1], out_height)  # Note: top to bottom for image coordinates
        
        return data, lons, lats, bounds


def raster_to_png_data_uri(data):
    """Convert raster array to a PNG data URI for Mapbox image layer.
    
    Nodata values (NaN) are rendered as transparent.

    Returns:
        tuple: (data_uri, vmin, vmax)
    """
    data = data.astype(np.float32)
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        raise ValueError("No valid data in raster")

    vmin = np.nanpercentile(data, 2)
    vmax = np.nanpercentile(data, 98)
    scaled = np.zeros_like(data, dtype=np.uint8)
    scaled[valid_mask] = np.clip((data[valid_mask] - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

    # Create RGBA with alpha channel for transparency
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    rgba[:, :, 0] = scaled  # Red channel
    rgba[:, :, 1] = scaled  # Green channel
    rgba[:, :, 2] = scaled  # Blue channel
    rgba[:, :, 3] = np.where(valid_mask, 255, 0)  # Alpha: 255 for valid, 0 for nodata
    
    img = Image.fromarray(rgba, mode='RGBA')
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}", float(vmin), float(vmax)


def create_mapbox_raster_figure(raster_path, title, mapbox_token, colorscale="Gray"):
    """Create a Plotly figure with raster overlay on Mapbox satellite basemap.
    
    Args:
        raster_path: Path to the raster file
        title: Title for the figure
        mapbox_token: Mapbox access token
        colorscale: Plotly colorscale for the colorbar
        
    Returns:
        Plotly figure object
    """
    dsm_data, _, _, bounds = get_raster_data(raster_path)
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2

    image_uri, vmin, vmax = raster_to_png_data_uri(dsm_data)
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

    fig = px.scatter_mapbox(
        lat=[center_lat],
        lon=[center_lon],
        zoom=17,
    )
    # Add a transparent scattermapbox trace to show a colorbar
    fig.add_trace(
        go.Scattermapbox(
            lat=[center_lat, center_lat],
            lon=[center_lon, center_lon],
            mode="markers",
            marker=dict(
                size=1,
                opacity=0,
                color=[vmin, vmax],
                colorscale=colorscale,
                cmin=vmin,
                cmax=vmax,
                showscale=True,
                colorbar=dict(title="Elevation (m)"),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        mapbox_style="satellite",
        mapbox_accesstoken=mapbox_token,
        mapbox_layers=[image_layer],
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        title=title,
    )
    
    return fig
