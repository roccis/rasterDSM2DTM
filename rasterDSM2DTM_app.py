import streamlit as st
import tempfile
import os
from raster_utils import (
    dsm_to_dtm_metric,
    create_mapbox_raster_figure
)

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

# Upload options
st.subheader("📤 Upload DSM")
uploaded_file = st.file_uploader(
    "Upload DSM GeoTIFF",
    type=['tif', 'tiff'],
    help="Upload a Digital Surface Model in GeoTIFF format"
)

# Only show processing if we have a file
can_process = uploaded_file is not None

if can_process:
    # Process button
    if st.button("🚀 Process DSM", type="primary"):
        # Store processing state
        st.session_state.processing = True
    
    if st.session_state.get('processing', False):
        progress_bar = st.progress(0, "Preparing...")
        try:
            # Create temporary files and load data
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_input:
                tmp_input.write(uploaded_file.getvalue())
                input_path = tmp_input.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_output:
                output_path = tmp_output.name

            # Process the DSM
            progress_bar.progress(30, f"🔄 Processing DSM (window={window_size}m)... This may take several minutes for large files.")
            dtm_path, chm_path, metadata = dsm_to_dtm_metric(
                input_path, 
                output_path, 
                search_radius_meters=window_size
            )
            
            progress_bar.progress(80, "🗺️ Generating visualizations...")
            
            # Store results in session state to persist across reruns
            st.session_state.processing_results = {
                'input_path': input_path,
                'dtm_path': dtm_path,
                'chm_path': chm_path,
                'metadata': metadata
            }
            
            progress_bar.progress(100, "✅ Complete!")

            # Display metadata
            st.success("✅ Processing complete!")
            
            if metadata.get('downsampled', False):
                st.warning(f"⚠️ Large raster was automatically downsampled by {metadata['downsample_factor']}x for processing to prevent memory issues. Original: {metadata['original_shape'][1]}x{metadata['original_shape'][0]} pixels, Processed: {metadata['shape'][1]}x{metadata['shape'][0]} pixels.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Resolution", f"{metadata['resolution']:.2f}m")
            with col2:
                st.metric("Window (pixels)", metadata['window_pixels'])
            with col3:
                st.metric("Window (meters)", f"{metadata['window_meters']:.1f}m")
            
            # Visualization section
            st.header("📊 Results Visualization")
            st.info("Large rasters are automatically downsampled for display performance.")
            
            tab1, tab2, tab3 = st.tabs(["📍 DSM", "🏔️ DTM", "🌳 CHM"])

            with tab1:
                fig = create_mapbox_raster_figure(input_path, "DSM", MAPBOX_TOKEN)
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

            with tab2:
                fig = create_mapbox_raster_figure(dtm_path, "DTM", MAPBOX_TOKEN)
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

            with tab3:
                fig = create_mapbox_raster_figure(chm_path, "CHM", MAPBOX_TOKEN)
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
            
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
            if 'input_path' in locals() and os.path.exists(input_path):
                os.unlink(input_path)
            if 'output_path' in locals() and os.path.exists(output_path):
                os.unlink(output_path)
            if 'progress_bar' in locals():
                progress_bar.empty()
    
    elif st.session_state.get('processing_results'):
        # Show cached results if available
        results = st.session_state.processing_results
        metadata = results['metadata']
        input_path = results['input_path']
        dtm_path = results['dtm_path']
        chm_path = results['chm_path']
        
        st.success("✅ Processing complete!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Resolution", f"{metadata['resolution']:.2f}m")
        with col2:
            st.metric("Window (pixels)", metadata['window_pixels'])
        with col3:
            st.metric("Window (meters)", f"{metadata['window_meters']:.1f}m")
        
        # Visualization section
        st.header("📊 Results Visualization")
        st.info("Large rasters are automatically downsampled for display performance.")
        
        tab1, tab2, tab3 = st.tabs(["📍 DSM", "🏔️ DTM", "🌳 CHM"])

        with tab1:
            fig = create_mapbox_raster_figure(input_path, "DSM", MAPBOX_TOKEN)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

        with tab2:
            fig = create_mapbox_raster_figure(dtm_path, "DTM", MAPBOX_TOKEN)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})

        with tab3:
            fig = create_mapbox_raster_figure(chm_path, "CHM", MAPBOX_TOKEN)
            st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
        
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
    st.info("👆 Upload a DSM GeoTIFF file to get started")