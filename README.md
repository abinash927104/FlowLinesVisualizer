# 🌊 Flow Line Visualizer

An interactive Streamlit application for visualizing and analyzing streamlines, streaklines, and pathlines in fluid flow. The app includes computational flow visualizations and experimental image analysis capabilities.

## Features

- **Interactive Flow Visualizations**: Generate and visualize streamlines, streaklines, and pathlines for various flow types
- **Experiment Guide**: Step-by-step instructions for conducting flow visualization experiments with dye
- **Image Analysis**: Upload experimental images and automatically detect flow patterns using computer vision
- **Pattern Recognition**: Extract flow characteristics from captured images using image processing algorithms

## Installation

1. Clone or download this repository

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. The app will open in your default web browser at `http://localhost:8501`

3. Navigate through the different sections:
   - **Flow Visualizations**: Interactive theoretical flow patterns
   - **Experiment Guide**: Instructions for conducting experiments
   - **Image Analysis**: Upload and analyze experimental images
   - **About**: Information about flow lines and the application

## Experiment Procedure

### Materials Needed
- Clear container (glass tank or transparent box)
- Water
- Neutrally buoyant dye (food coloring works well)
- Flow source (pump, stirrer, or manual movement)
- Phone camera or digital camera
- Good lighting

### Steps

1. **Setup**: Fill container with water and position camera
2. **Create Flow**: Use stirring, pumping, or gravity to create flow
3. **Add Dye**: Inject dye at specific points for different flow line types
4. **Capture Images**: Take high-resolution photos with fast shutter speed
5. **Analyze**: Upload images to the Image Analysis section

### Camera Settings Recommendations
- ISO: 400-800
- Shutter Speed: 1/500s or faster
- Aperture: f/4 to f/8
- White Balance: Set manually

## Flow Types Available

- **Uniform Flow**: Constant velocity field
- **Source Flow**: Radial outward flow from a point
- **Vortex Flow**: Circular flow around a point
- **Doublet Flow**: Combined source and sink
- **Channel Flow**: Flow through a channel
- **Combined Flow**: Mix of different flow types

## Image Analysis Features

The image analysis module uses:
- **Edge Detection**: Canny edge detection to identify dye boundaries
- **Contour Analysis**: Detects and extracts flow patterns
- **Gradient Analysis**: Determines flow direction and magnitude
- **Pattern Extraction**: Identifies streamlines, pathlines, and streaklines

## Technical Details

### Dependencies
- Streamlit: Web application framework
- Plotly: Interactive visualizations
- OpenCV: Image processing and computer vision
- NumPy: Numerical computations
- Matplotlib: Static plotting
- TensorFlow: Deep learning capabilities (for future enhancements)

## Understanding Flow Lines

- **Streamlines**: Instantaneous flow direction at a given time
- **Pathlines**: Actual path followed by a single particle over time
- **Streaklines**: Locus of particles that have passed through a specific point

## Safety Tips

- Work in a well-ventilated area
- Use non-toxic dyes (food coloring is safe)
- Clean up spills immediately
- Dispose of dyed water properly

## Future Enhancements

- Deep learning models for enhanced pattern recognition
- Video analysis capabilities
- Quantitative flow measurements
- Export functionality for analysis results
- 3D flow visualization

## License

This project is open source and available for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or suggestions, please open an issue in the repository.

