import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Flow Line Visualizer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for interactive background
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    .main-header {
        text-align: center;
        color: white;
        padding: 2rem;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .experiment-step {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: #1f1f1f !important;
    }
    .experiment-step h3, .experiment-step h4 {
        color: #1f1f1f !important;
    }
    .experiment-step p, .experiment-step li, .experiment-step ol, .experiment-step ul {
        color: #1f1f1f !important;
    }
    .experiment-step strong {
        color: #1f1f1f !important;
    }
    .info-box {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: #1f1f1f !important;
    }
    .info-box h3, .info-box h4 {
        color: #1f1f1f !important;
    }
    .info-box p, .info-box li, .info-box ul {
        color: #1f1f1f !important;
    }
    .info-box strong {
        color: #1f1f1f !important;
    }
    /* Sidebar styling - ensure text is visible */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.95) !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #1f1f1f !important;
    }
    /* Sidebar radio buttons and labels */
    section[data-testid="stSidebar"] [data-testid="stRadio"] label,
    section[data-testid="stSidebar"] .stRadio label {
        color: #1f1f1f !important;
    }
    /* Ensure all Streamlit text elements have proper contrast - but exclude sidebar */
    .main .stMarkdown, .main .stMarkdown p, .main .stMarkdown li, .main .stMarkdown ul, .main .stMarkdown ol {
        color: #1f1f1f !important;
    }
    /* Headers - make sure they're visible on white backgrounds - but exclude sidebar */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: #1f1f1f !important;
    }
    /* Streamlit header elements */
    [data-testid="stHeader"] h1, [data-testid="stHeader"] h2, [data-testid="stHeader"] h3 {
        color: #1f1f1f !important;
    }
    /* Main content area text */
    .main .block-container {
        color: #1f1f1f !important;
    }
    /* Override Streamlit's default text color in content areas - but exclude sidebar */
    .main div[data-testid="stMarkdownContainer"] p,
    .main div[data-testid="stMarkdownContainer"] li,
    .main div[data-testid="stMarkdownContainer"] ul,
    .main div[data-testid="stMarkdownContainer"] ol {
        color: #1f1f1f !important;
    }
    /* Metric text */
    .main [data-testid="stMetricValue"], .main [data-testid="stMetricLabel"] {
        color: #1f1f1f !important;
    }
    /* Selectbox, slider labels - but exclude sidebar */
    .main label, .main .stSelectbox label, .main .stSlider label {
        color: #1f1f1f !important;
    }
    /* Checkbox labels - but exclude sidebar */
    .main .stCheckbox label {
        color: #1f1f1f !important;
    }
    /* File uploader text */
    .main .stFileUploader label {
        color: #1f1f1f !important;
    }
    /* All text in main content */
    .main .element-container {
        color: #1f1f1f !important;
    }
    .main .element-container p,
    .main .element-container li,
    .main .element-container span {
        color: #1f1f1f !important;
    }
    /* Info and success boxes */
    .main .stInfo, .main .stSuccess, .main .stWarning, .main .stError {
        color: #1f1f1f !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header"><h1>🌊 Flow Line Visualizer</h1><p>Interactive Visualization of Streamlines, Streaklines, and Pathlines</p></div>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a section:",
    ["📊 Flow Visualizations", "🧪 Experiment Guide", "📸 Image Analysis", "ℹ️ About"]
)

# Flow visualization functions
def generate_velocity_field(x, y, flow_type="uniform"):
    """Generate velocity field for different flow types"""
    if flow_type == "uniform":
        u = np.ones_like(x) * 2.0
        v = np.zeros_like(y)
    elif flow_type == "source":
        r = np.sqrt(x**2 + y**2)
        u = x / (r + 0.1)
        v = y / (r + 0.1)
    elif flow_type == "vortex":
        r = np.sqrt(x**2 + y**2)
        u = -y / (r**2 + 0.1)
        v = x / (r**2 + 0.1)
    elif flow_type == "doublet":
        r = np.sqrt(x**2 + y**2)
        u = (x**2 - y**2) / (r**4 + 0.1)
        v = 2 * x * y / (r**4 + 0.1)
    elif flow_type == "channel":
        u = 1 - (y/5)**2
        v = np.zeros_like(y)
    else:  # combined
        r = np.sqrt(x**2 + y**2)
        u = 1.0 + x / (r + 0.1) - y / (r**2 + 0.1)
        v = y / (r + 0.1) + x / (r**2 + 0.1)
    return u, v

def compute_streamlines(x, y, u, v, start_points):
    """Compute streamlines from velocity field"""
    streamlines = []
    dt = 0.1
    max_steps = 200
    
    for start in start_points:
        line = [start]
        current = np.array(start, dtype=float)
        
        for _ in range(max_steps):
            # Interpolate velocity at current position
            i = int((current[0] - x[0, 0]) / (x[0, 1] - x[0, 0]))
            j = int((current[1] - y[0, 0]) / (y[1, 0] - y[0, 0]))
            
            if 0 <= i < u.shape[1] and 0 <= j < u.shape[0]:
                vel = np.array([u[j, i], v[j, i]])
                current += vel * dt
                line.append(current.copy())
                
                if np.abs(current[0]) > 10 or np.abs(current[1]) > 10:
                    break
            else:
                break
        
        streamlines.append(np.array(line))
    
    return streamlines

def compute_pathline(x, y, u, v, start_point, time_steps):
    """Compute pathline for a particle"""
    pathline = [start_point]
    current = np.array(start_point, dtype=float)
    dt = 0.1
    
    for _ in range(time_steps):
        i = int((current[0] - x[0, 0]) / (x[0, 1] - x[0, 0]))
        j = int((current[1] - y[0, 0]) / (y[1, 0] - y[0, 0]))
        
        if 0 <= i < u.shape[1] and 0 <= j < u.shape[0]:
            vel = np.array([u[j, i], v[j, i]])
            current += vel * dt
            pathline.append(current.copy())
            
            if np.abs(current[0]) > 10 or np.abs(current[1]) > 10:
                break
        else:
            break
    
    return np.array(pathline)

def compute_streakline(x, y, u, v, injection_point, num_particles, time_steps):
    """Compute streakline from continuous injection"""
    streakline = []
    particles = []
    
    for t in range(time_steps):
        # Add new particle at injection point
        if t % 5 == 0:  # Inject every 5 time steps
            particles.append({
                'pos': np.array(injection_point, dtype=float),
                'age': 0
            })
        
        # Update all particles
        for particle in particles:
            pos = particle['pos']
            i = int((pos[0] - x[0, 0]) / (x[0, 1] - x[0, 0]))
            j = int((pos[1] - y[0, 0]) / (y[1, 0] - y[0, 0]))
            
            if 0 <= i < u.shape[1] and 0 <= j < u.shape[0]:
                vel = np.array([u[j, i], v[j, i]])
                particle['pos'] += vel * 0.1
                particle['age'] += 1
        
        # Remove old particles
        particles = [p for p in particles if p['age'] < num_particles and 
                    np.abs(p['pos'][0]) < 10 and np.abs(p['pos'][1]) < 10]
    
    # Collect positions for streakline
    for particle in particles:
        streakline.append(particle['pos'])
    
    return np.array(streakline) if streakline else np.array([])

# Image processing functions
def detect_flow_patterns(image):
    """Detect flow patterns using optical flow and image processing"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Detect contours (dye patterns)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area
    min_area = 100
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # Detect flow direction using gradient
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x)
    
    return {
        'edges': edges,
        'contours': filtered_contours,
        'magnitude': magnitude,
        'angle': angle,
        'grad_x': grad_x,
        'grad_y': grad_y
    }

def extract_flow_lines(contours, angle, magnitude, threshold=0.3):
    """Extract flow lines from detected patterns"""
    flow_lines = []
    
    for contour in contours:
        if len(contour) < 5:
            continue
        
        # Get points along contour
        points = contour.reshape(-1, 2)
        
        # Calculate flow direction at each point
        line_points = []
        for point in points[::2]:  # Sample every other point
            y, x = int(point[1]), int(point[0])
            if 0 <= y < angle.shape[0] and 0 <= x < angle.shape[1]:
                if magnitude[y, x] > threshold * magnitude.max():
                    line_points.append({
                        'x': x,
                        'y': y,
                        'angle': angle[y, x],
                        'magnitude': magnitude[y, x]
                    })
        
        if line_points:
            flow_lines.append(line_points)
    
    return flow_lines

# Main app logic
if page == "📊 Flow Visualizations":
    st.header("Interactive Flow Visualizations")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Flow Parameters")
        flow_type = st.selectbox(
            "Flow Type",
            ["uniform", "source", "vortex", "doublet", "channel", "combined"]
        )
        
        num_streamlines = st.slider("Number of Streamlines", 5, 30, 10)
        streamline_length = st.slider("Streamline Length", 50, 500, 200)
        
        show_pathline = st.checkbox("Show Pathline", value=True)
        show_streakline = st.checkbox("Show Streakline", value=True)
        
        if show_pathline:
            pathline_start_x = st.slider("Pathline Start X", -5.0, 5.0, 0.0)
            pathline_start_y = st.slider("Pathline Start Y", -5.0, 5.0, 0.0)
        
        if show_streakline:
            streakline_inject_x = st.slider("Streakline Injection X", -5.0, 5.0, -2.0)
            streakline_inject_y = st.slider("Streakline Injection Y", -5.0, 5.0, 0.0)
    
    with col2:
        # Generate velocity field
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        U, V = generate_velocity_field(X, Y, flow_type)
        
        # Create figure
        fig = go.Figure()
        
        # Add velocity field as quiver plot
        fig.add_trace(go.Scatter(
            x=X.flatten()[::5],
            y=Y.flatten()[::5],
            mode='markers',
            marker=dict(size=1, color='lightblue', opacity=0.3),
            showlegend=False
        ))
        
        # Compute and plot streamlines
        start_points = [(np.random.uniform(-4, 4), np.random.uniform(-4, 4)) 
                       for _ in range(num_streamlines)]
        streamlines = compute_streamlines(X, Y, U, V, start_points)
        
        for i, line in enumerate(streamlines):
            fig.add_trace(go.Scatter(
                x=line[:, 0],
                y=line[:, 1],
                mode='lines',
                name=f'Streamline {i+1}' if i < 5 else '',
                line=dict(color='blue', width=2),
                showlegend=(i < 5)
            ))
        
        # Compute and plot pathline
        if show_pathline:
            pathline = compute_pathline(X, Y, U, V, (pathline_start_x, pathline_start_y), streamline_length)
            fig.add_trace(go.Scatter(
                x=pathline[:, 0],
                y=pathline[:, 1],
                mode='lines+markers',
                name='Pathline',
                line=dict(color='red', width=3),
                marker=dict(size=5, color='red')
            ))
        
        # Compute and plot streakline
        if show_streakline:
            streakline = compute_streakline(X, Y, U, V, (streakline_inject_x, streakline_inject_y), 
                                           streamline_length, streamline_length)
            if len(streakline) > 0:
                fig.add_trace(go.Scatter(
                    x=streakline[:, 0],
                    y=streakline[:, 1],
                    mode='lines+markers',
                    name='Streakline',
                    line=dict(color='green', width=3),
                    marker=dict(size=4, color='green')
                ))
        
        fig.update_layout(
            title="Flow Visualization: Streamlines, Pathlines, and Streaklines",
            xaxis_title="X",
            yaxis_title="Y",
            width=800,
            height=600,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Explanation
        st.markdown("""
        <div class="info-box">
        <h4>Understanding the Visualizations:</h4>
        <ul>
        <li><strong>Streamlines (Blue):</strong> Instantaneous flow direction at a given time</li>
        <li><strong>Pathline (Red):</strong> Actual path followed by a single particle over time</li>
        <li><strong>Streakline (Green):</strong> Locus of particles that have passed through a specific point</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

elif page == "🧪 Experiment Guide":
    st.header("Experiment Guide: Flow Visualization with Dye")
    
    st.markdown("""
    <div class="experiment-step">
    <h3>Step 1: Preparation</h3>
    <p><strong>Materials Needed:</strong></p>
    <ul>
    <li>Clear container (glass tank or transparent box)</li>
    <li>Water</li>
    <li>Neutrally buoyant dye (food coloring or specialized flow visualization dye)</li>
    <li>Flow source (pump, stirrer, or manual movement)</li>
    <li>Phone camera or digital camera</li>
    <li>Good lighting (preferably from the side or back)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="experiment-step">
    <h3>Step 2: Setup</h3>
    <ol>
    <li>Fill the container with water (leave some space at the top)</li>
    <li>Ensure the container is clean and free of bubbles</li>
    <li>Set up your camera on a tripod or stable surface</li>
    <li>Position lighting to highlight the dye against the background</li>
    <li>Test camera focus and exposure settings</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="experiment-step">
    <h3>Step 3: Creating Flow</h3>
    <p>Choose one of these methods to create flow:</p>
    <ul>
    <li><strong>Stirring:</strong> Use a rod or spoon to create circular flow</li>
    <li><strong>Pump:</strong> Use a small pump to create directional flow</li>
    <li><strong>Gravity:</strong> Pour water from one side to create flow</li>
    <li><strong>Moving Object:</strong> Move an object through the water</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="experiment-step">
    <h3>Step 4: Adding Dye</h3>
    <ol>
    <li>Wait for the flow to stabilize</li>
    <li>Carefully inject a small amount of neutrally buoyant dye at a specific point</li>
    <li>For <strong>Pathline:</strong> Inject dye and track a single particle</li>
    <li>For <strong>Streakline:</strong> Continuously inject dye at the same point</li>
    <li>For <strong>Streamline:</strong> Take a snapshot of the dye pattern at one instant</li>
    </ol>
    <p><strong>Tip:</strong> Use a syringe or dropper for precise dye injection</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="experiment-step">
    <h3>Step 5: Capturing Images</h3>
    <ol>
    <li>Set camera to high resolution mode</li>
    <li>Use a fast shutter speed to freeze motion (1/500s or faster)</li>
    <li>Take multiple photos at different time intervals</li>
    <li>Ensure good contrast between dye and water</li>
    <li>Capture from a consistent angle</li>
    </ol>
    <p><strong>Camera Settings Recommendations:</strong></p>
    <ul>
    <li>ISO: 400-800 (adjust based on lighting)</li>
    <li>Shutter Speed: 1/500s or faster</li>
    <li>Aperture: f/4 to f/8 for good depth of field</li>
    <li>White Balance: Set manually for consistent colors</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="experiment-step">
    <h3>Step 6: Image Analysis</h3>
    <p>Upload your captured images to the <strong>Image Analysis</strong> section to:</p>
    <ul>
    <li>Detect flow patterns automatically</li>
    <li>Extract streamlines, pathlines, and streaklines</li>
    <li>Analyze flow direction and velocity</li>
    <li>Visualize flow characteristics</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>Safety Tips:</h4>
    <ul>
    <li>Work in a well-ventilated area</li>
    <li>Use non-toxic dyes (food coloring is safe)</li>
    <li>Clean up spills immediately</li>
    <li>Dispose of dyed water properly</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif page == "📸 Image Analysis":
    st.header("Flow Pattern Recognition from Images")
    
    st.markdown("""
    <div class="info-box">
    <p>Upload an image of your flow experiment with dye. The app will automatically detect and analyze flow patterns using image processing and computer vision techniques.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image captured from your flow visualization experiment"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Processed Image")
            
            # Process image
            processed = detect_flow_patterns(image_array)
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            
            # Original
            axes[0, 0].imshow(image_array)
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis('off')
            
            # Edges
            axes[0, 1].imshow(processed['edges'], cmap='gray')
            axes[0, 1].set_title("Edge Detection")
            axes[0, 1].axis('off')
            
            # Magnitude
            axes[1, 0].imshow(processed['magnitude'], cmap='hot')
            axes[1, 0].set_title("Flow Magnitude")
            axes[1, 0].axis('off')
            
            # Contours
            contour_img = image_array.copy()
            cv2.drawContours(contour_img, processed['contours'], -1, (0, 255, 0), 2)
            axes[1, 1].imshow(contour_img)
            axes[1, 1].set_title(f"Detected Patterns ({len(processed['contours'])} contours)")
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Extract flow lines
        flow_lines = extract_flow_lines(processed['contours'], processed['angle'], processed['magnitude'])
        
        st.subheader("Flow Analysis Results")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.metric("Detected Contours", len(processed['contours']))
            st.metric("Flow Lines Extracted", len(flow_lines))
            st.metric("Average Flow Magnitude", f"{processed['magnitude'].mean():.2f}")
        
        with col4:
            # Flow direction visualization
            fig2 = plt.figure(figsize=(10, 8))
            plt.imshow(image_array)
            
            # Draw flow vectors
            step = 20
            h, w = processed['angle'].shape
            y_coords = np.arange(0, h, step)
            x_coords = np.arange(0, w, step)
            X, Y = np.meshgrid(x_coords, y_coords)
            
            U = np.cos(processed['angle'][::step, ::step])
            V = np.sin(processed['angle'][::step, ::step])
            M = processed['magnitude'][::step, ::step]
            
            # Normalize for visualization
            U = U * M / M.max() * step
            V = V * M / M.max() * step
            
            plt.quiver(X, Y, U, V, M, cmap='viridis', scale=50, alpha=0.7)
            plt.title("Flow Direction Vectors")
            plt.axis('off')
            st.pyplot(fig2)
        
        # Flow line details
        if flow_lines:
            st.subheader("Extracted Flow Lines")
            
            # Create interactive plot
            fig3 = go.Figure()
            
            # Add original image as background
            fig3.add_trace(go.Scatter(
                x=[0, image_array.shape[1]],
                y=[0, image_array.shape[0]],
                mode='markers',
                marker=dict(size=1, opacity=0),
                showlegend=False
            ))
            
            # Add flow lines
            colors = px.colors.qualitative.Set3
            for i, line in enumerate(flow_lines[:10]):  # Limit to first 10 for performance
                if line:
                    xs = [p['x'] for p in line]
                    ys = [p['y'] for p in line]
                    fig3.add_trace(go.Scatter(
                        x=xs,
                        y=ys,
                        mode='lines+markers',
                        name=f'Flow Line {i+1}',
                        line=dict(color=colors[i % len(colors)], width=2),
                        marker=dict(size=3)
                    ))
            
            fig3.update_layout(
                title="Detected Flow Lines",
                xaxis=dict(scaleanchor="y", scaleratio=1),
                yaxis=dict(autorange='reversed'),
                width=800,
                height=600,
                template="plotly_white"
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        
        # Download results
        st.subheader("Export Results")
        st.info("Flow pattern analysis complete! The detected patterns can be used to understand streamline, streakline, and pathline characteristics in your experiment.")

elif page == "ℹ️ About":
    st.header("About Flow Line Visualizer")
    
    st.markdown("""
    <div class="info-box">
    <h3>What are Flow Lines?</h3>
    <p>Flow lines are different ways to visualize and understand fluid motion:</p>
    <ul>
    <li><strong>Streamlines:</strong> Curves that are tangent to the velocity vector field at every point. They represent the instantaneous flow direction.</li>
    <li><strong>Pathlines:</strong> The actual trajectory that a fluid particle follows over time. This is what you see when tracking a single particle.</li>
    <li><strong>Streaklines:</strong> The locus of all fluid particles that have passed through a particular point. This is what you see when continuously injecting dye at one location.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>How It Works</h3>
    <p>This app uses:</p>
    <ul>
    <li><strong>Computational Fluid Dynamics:</strong> To generate and visualize theoretical flow patterns</li>
    <li><strong>Computer Vision:</strong> Edge detection and contour analysis to identify dye patterns</li>
    <li><strong>Image Processing:</strong> Gradient analysis to determine flow direction and magnitude</li>
    <li><strong>Pattern Recognition:</strong> To extract and classify flow characteristics from experimental images</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>Applications</h3>
    <ul>
    <li>Understanding fluid dynamics concepts</li>
    <li>Analyzing flow patterns in experiments</li>
    <li>Educational demonstrations</li>
    <li>Research and development</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>Technical Details</h3>
    <p>Built with:</p>
    <ul>
    <li>Streamlit for the web interface</li>
    <li>Plotly for interactive visualizations</li>
    <li>OpenCV for image processing</li>
    <li>NumPy for numerical computations</li>
    <li>Matplotlib for static plots</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

