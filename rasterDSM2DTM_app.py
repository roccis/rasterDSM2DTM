import streamlit as st
import tempfile
import os
import s3fs
import boto3
import uuid
from datetime import datetime, timezone
import streamlit.components.v1 as components
from raster_utils import (
    dsm_to_dtm_metric,
    create_mapbox_raster_figure
)

s3_client = boto3.client('s3',aws_access_key_id = st.secrets["aws"]["access_key"], aws_secret_access_key = st.secrets["aws"]["secret_key"],region_name = st.secrets["aws"]["region"])
fs = s3fs.S3FileSystem(key=st.secrets["aws"]["access_key"], secret=st.secrets["aws"]["secret_key"])
bucket_name = st.secrets["aws"]["bucket_name"]
bucket_prefix = st.secrets["aws"].get("prefix", "temp_rasters")
expiry_days = int(st.secrets["aws"].get("expiry_days", 1))


def ensure_lifecycle_rule(bucket, prefix, days):
    rule_id = "rasterdsm2dtm-expire"
    rule = {
        "ID": rule_id,
        "Filter": {"Prefix": f"{prefix}/"},
        "Status": "Enabled",
        "Expiration": {"Days": days},
    }
    try:
        existing = s3_client.get_bucket_lifecycle_configuration(Bucket=bucket)
        rules = existing.get("Rules", [])
        if not any(r.get("ID") == rule_id for r in rules):
            rules.append(rule)
            s3_client.put_bucket_lifecycle_configuration(
                Bucket=bucket,
                LifecycleConfiguration={"Rules": rules},
            )
    except s3_client.exceptions.NoSuchBucket:
        st.error(f"Bucket {bucket} does not exist")
    except Exception as e:
        # Silently skip lifecycle configuration if permissions denied
        # This is optional and doesn't affect app functionality
        error_code = e.response.get('Error', {}).get('Code', '') if hasattr(e, 'response') else ''
        if error_code in ['NoSuchLifecycleConfiguration', 'AccessDenied']:
            pass  # Skip silently - lifecycle config is optional
        else:
            st.warning(f"Could not set lifecycle rule: {e}")


def upload_to_s3(local_path, key, content_type="application/octet-stream"):
    try:
        # Try with tagging first
        tagging = f"app=rasterdsm2dtm&delete_after_days={expiry_days}"
        s3_client.upload_file(
            local_path,
            bucket_name,
            key,
            ExtraArgs={
                "ContentType": content_type,
                "Tagging": tagging,
            },
        )
    except Exception as e:
        # If tagging fails (permission denied), upload without tags
        if "PutObjectTagging" in str(e) or "AccessDenied" in str(e):
            s3_client.upload_file(
                local_path,
                bucket_name,
                key,
                ExtraArgs={
                    "ContentType": content_type,
                },
            )
        else:
            raise e
    return f"s3://{bucket_name}/{key}"


def presigned_url(key, expires=3600):
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket_name, "Key": key},
        ExpiresIn=expires,
    )


def presigned_post(key, expires=900):
    return s3_client.generate_presigned_post(
        Bucket=bucket_name,
        Key=key,
        ExpiresIn=expires,
        Conditions=[
            ["content-length-range", 1, 10_000_000_000],
        ],
    )


def download_from_s3(key, local_path):
    s3_client.download_file(bucket_name, key, local_path)


def check_s3_object_exists(key):
    """Check if an object exists in S3"""
    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
        return True
    except:
        return False

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
upload_mode = st.radio(
    "Choose upload method",
    ["Direct to S3 (no 200MB limit)", "Upload via Streamlit"],
    index=0,
)

uploaded_file = None
direct_s3_key = None

if upload_mode == "Upload via Streamlit":
    uploaded_file = st.file_uploader(
        "Upload DSM GeoTIFF",
        type=['tif', 'tiff'],
        help="Upload a Digital Surface Model in GeoTIFF format"
    )
else:
    if "direct_s3_key" not in st.session_state:
        session_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        st.session_state.direct_s3_key = f"{bucket_prefix}/{session_id}/dsm.tif"

    direct_s3_key = st.session_state.direct_s3_key
    presigned = presigned_post(direct_s3_key)

    st.markdown("**Upload directly to S3 (bypasses Streamlit size limit):**")
    
    upload_html = f"""
    <style>
        .upload-container {{
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            background: #f9f9f9;
        }}
        .upload-status {{
            margin-top: 15px;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
        }}
        .status-waiting {{ background: #fff3cd; color: #856404; }}
        .status-uploading {{ background: #d1ecf1; color: #0c5460; }}
        .status-success {{ background: #d4edda; color: #155724; }}
        .status-error {{ background: #f8d7da; color: #721c24; }}
        input[type="file"] {{
            margin: 10px 0;
            padding: 10px;
        }}
        input[type="submit"] {{
            background: #0066cc;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }}
        input[type="submit"]:hover {{
            background: #0052a3;
        }}
        input[type="submit"]:disabled {{
            background: #ccc;
            cursor: not-allowed;
        }}
    </style>
    <div class="upload-container">
        <form id="s3-upload-form" action="{presigned['url']}" method="post" enctype="multipart/form-data">
            {''.join([f'<input type="hidden" name="{k}" value="{v}">' for k, v in presigned['fields'].items()])}
            <div>
                <label for="file-input" style="font-size: 16px;">📁 Choose GeoTIFF file:</label><br>
                <input type="file" id="file-input" name="file" accept=".tif,.tiff" required />
            </div>
            <input type="submit" id="submit-btn" value="⬆️ Upload to S3" />
        </form>
        <div id="status" class="upload-status status-waiting">
            📋 Select a file and click Upload to S3
        </div>
    </div>
    <script>
        const form = document.getElementById('s3-upload-form');
        const fileInput = document.getElementById('file-input');
        const submitBtn = document.getElementById('submit-btn');
        const status = document.getElementById('status');
        
        fileInput.addEventListener('change', function() {{
            if (this.files.length > 0) {{
                const fileName = this.files[0].name;
                const fileSize = (this.files[0].size / 1024 / 1024).toFixed(2);
                status.className = 'upload-status status-waiting';
                status.innerHTML = `📄 Selected: ${{fileName}} (${{fileSize}} MB)<br>Click Upload to S3 to continue`;
            }}
        }});
        
        form.addEventListener('submit', function(e) {{
            submitBtn.disabled = true;
            status.className = 'upload-status status-uploading';
            status.innerHTML = '⏳ Uploading to S3... Please wait';
        }});
        
        // Check for redirect back (upload complete)
        if (window.location.href.indexOf('key=') > -1) {{
            status.className = 'upload-status status-success';
            status.innerHTML = '✅ Upload complete! Verifying file availability...';
            // Wait longer for S3 to finalize large files
            setTimeout(function() {{
                window.location.href = window.location.pathname + '?uploaded=true';
            }}, 3000);
        }}
    </script>
    """
    components.html(upload_html, height=280)

    st.divider()
    
    # Check if file was uploaded
    # Add query param check for post-upload verification
    query_params = st.query_params
    if query_params.get("uploaded") == "true":
        with st.spinner("🔍 Verifying file in S3..."):
            import time
            # Poll for file existence (S3 eventual consistency)
            max_attempts = 10
            for attempt in range(max_attempts):
                if check_s3_object_exists(direct_s3_key):
                    st.success(f"✅ **File ready in S3!** Click 'Process DSM' below to continue.")
                    # Clear the query param
                    st.query_params.clear()
                    break
                time.sleep(1)
            else:
                st.error("❌ Could not verify file upload. Please try again or check your S3 bucket.")
    elif check_s3_object_exists(direct_s3_key):
        st.success(f"✅ **File ready in S3!** Click 'Process DSM' below to continue.")
    else:
        st.info("⏳ Waiting for file upload...")

# Only show processing if we have a file
can_process = uploaded_file is not None or (direct_s3_key is not None and check_s3_object_exists(direct_s3_key))

if can_process:
    # Process button
    if st.button("🚀 Process DSM", type="primary"):
        progress_bar = st.progress(0, "Preparing...")
        try:
            # Create temporary files and load data
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_input:
                if uploaded_file is not None:
                    tmp_input.write(uploaded_file.getvalue())
                    input_path = tmp_input.name
                else:
                    input_path = tmp_input.name
            
            # Download from S3 if using direct upload
            if uploaded_file is None and direct_s3_key is not None:
                progress_bar.progress(10, "📥 Downloading file from S3...")
                download_from_s3(direct_s3_key, input_path)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_output:
                output_path = tmp_output.name
            
            progress_bar.progress(20, "⚙️ Setting up S3...")
            ensure_lifecycle_rule(bucket_name, bucket_prefix, expiry_days)
            session_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"

            # Process the DSM
            progress_bar.progress(30, f"🔄 Processing DSM (window={window_size}m)... This may take several minutes for large files.")
            dtm_path, chm_path, metadata = dsm_to_dtm_metric(
                input_path, 
                output_path, 
                search_radius_meters=window_size
            )

            # Upload original and outputs to S3
            progress_bar.progress(60, "☁️ Uploading results to S3...")
            dsm_key = direct_s3_key or f"{bucket_prefix}/{session_id}/dsm.tif"
            dtm_key = f"{bucket_prefix}/{session_id}/dtm.tif"
            chm_key = f"{bucket_prefix}/{session_id}/chm.tif"

            if uploaded_file is not None:
                dsm_s3 = upload_to_s3(input_path, dsm_key, content_type="image/tiff")
            else:
                dsm_s3 = f"s3://{bucket_name}/{dsm_key}"
            dtm_s3 = upload_to_s3(dtm_path, dtm_key, content_type="image/tiff")
            chm_s3 = upload_to_s3(chm_path, chm_key, content_type="image/tiff")
            
            progress_bar.progress(80, "🗺️ Generating visualizations...")
            
            # Display metadata
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
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                fig = create_mapbox_raster_figure(dtm_path, "DTM", MAPBOX_TOKEN)
                st.plotly_chart(fig, use_container_width=True)

            with tab3:
                fig = create_mapbox_raster_figure(chm_path, "CHM", MAPBOX_TOKEN)
                st.plotly_chart(fig, use_container_width=True)
            
            progress_bar.progress(100, "✅ Complete!")

            st.subheader("☁️ Stored in S3")
            st.write(f"DSM: {dsm_s3}")
            st.write(f"DTM: {dtm_s3}")
            st.write(f"CHM: {chm_s3}")

            st.subheader("🔗 Temporary download links")
            st.write(presigned_url(dsm_key))
            st.write(presigned_url(dtm_key))
            st.write(presigned_url(chm_key))
            
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
else:
    st.info("👆 Upload a DSM GeoTIFF file to get started")