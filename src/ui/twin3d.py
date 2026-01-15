"""
3D Patient Twin Component for Streamlit

Renders a wireframe 3D head visualization that responds to patient vitals.
"""

import streamlit.components.v1 as components
from pathlib import Path
import json
import base64


def render_twin3d(payload: dict, height: int = 460):
    """
    Render 3D Patient Twin component.
    
    Args:
        payload: Dictionary with keys:
            - gender (int): 1 for male, 0 for female
            - height_cm (float): Height in centimeters
            - weight_kg (float): Weight in kilograms
            - bmi (float): Body mass index
            - vl (float): Viral load (log copies/mL)
            - cd4 (float): CD4 count (cells/μL)
        height: Height of the iframe in pixels (default: 460)
    
    Returns:
        Streamlit component instance
    """
    # Get the absolute path to the web/twin directory
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    twin_dir = project_root / "web" / "twin"
    html_path = twin_dir / "index.html"
    
    if not html_path.exists():
        raise FileNotFoundError(
            f"3D Twin HTML not found at {html_path}. "
            "Make sure web/twin/index.html exists."
        )
    
    # Read HTML, CSS, and JS files to inline them
    html_content = html_path.read_text(encoding='utf-8')
    
    # Update asset paths to be relative to the HTML file location
    # When embedded in Streamlit, paths need to be relative to the HTML file
    html_content = html_content.replace('src="twin.js"', 'src="./twin.js"')
    
    # Read and inline CSS
    css_path = twin_dir / "styles.css"
    if css_path.exists():
        css_content = css_path.read_text(encoding='utf-8')
        html_content = html_content.replace(
            '<link rel="stylesheet" href="styles.css">',
            f'<style>{css_content}</style>'
        )
    
    # Read and inline JS
    js_path = twin_dir / "twin.js"
    if not js_path.exists():
        raise FileNotFoundError(f"twin.js not found at {js_path}")
    
    js_content = js_path.read_text(encoding='utf-8')
    
    # Load GLB files as base64 data URLs for embedding
    glb_data_urls = {}
    assets_dir = twin_dir / "assets"
    
    # Try to load GLB files as base64
    for glb_name in ['human_male.glb', 'human_female.glb']:
        glb_path = assets_dir / glb_name
        if glb_path.exists():
            try:
                with open(glb_path, 'rb') as f:
                    glb_data = f.read()
                    glb_base64 = base64.b64encode(glb_data).decode('utf-8')
                    glb_data_urls[glb_name] = f'data:model/gltf-binary;base64,{glb_base64}'
                    print(f'[Twin3D] Loaded {glb_name} as base64 ({len(glb_data)} bytes)')
            except Exception as e:
                print(f'[Twin3D] Warning: Failed to load {glb_name}: {e}')
    
    # Create boot script with payload and inline JS
    payload_json = json.dumps(payload)
    glb_data_urls_json = json.dumps(glb_data_urls)
    
    # Combine boot script with twin.js and GLB data URLs
    combined_script = f"""
    <script>
        console.log('[Twin3D Boot] Setting payload:', {payload_json});
        // Store payload globally for twin.js to access
        window.__twin_payload = {payload_json};
        
        // Store GLB data URLs for direct loading
        window.__twin_glb_data_urls = {glb_data_urls_json};
        
        // Set payload as data attribute for debugging
        if (document.body) {{
            document.body.setAttribute('data-payload', JSON.stringify(window.__twin_payload));
        }}
    </script>
    <script>
        {js_content}
    </script>
    <script>
        // Boot script to trigger update after everything loads
        window.addEventListener('load', function() {{
            console.log('[Twin3D Boot] Window loaded, payload:', window.__twin_payload);
            setTimeout(function() {{
                if (window.updateTwin && window.__twin_payload) {{
                    console.log('[Twin3D Boot] Updating on load');
                    window.updateTwin(window.__twin_payload);
                }}
            }}, 800);
        }});
    </script>
    """
    
    # Replace script tag with inlined version
    if '<script src="twin.js"></script>' in html_content:
        html_content = html_content.replace(
            '<script src="twin.js"></script>',
            combined_script
        )
    else:
        # If script tag format is different, append before </body>
        html_content = html_content.replace('</body>', f'{combined_script}</body>')
    
    # Render component with minimal spacing
    return components.html(
        html_content,
        height=height,
        scrolling=False
    )

