# MSoOHC
Making Sense of  Organizational Hierarchy Charts with AI for SNA

# Key Value Proposition

Automate key steps in organizational network analysis by analyzing formal hierarchies (from .png organizational charts). So they can be compared with informal knowledge flows (from social network analysis).


1. Automates Tedious Manual Work
2. Extracts structure from org charts (positions of actors, reporting lines)
2. Maps the position of actors in the organizational chart (people, teams, roles, department, etc)
4. Saves hours of manual chart annotation and alignment

# Inputs 

A .png image captuing the organizational chart position

# Outputs 

A JSON file with AI interpretation of where actors are positioned  on the image (on a two-dimentional plan x,y). 
This can be used to guide a Social Network Analysis to position nodes over the organizational chart (using  a organization chart as background image). 

# Organizational Chart Analyzer

A hybrid OpenCV + LLaVA solution for analyzing organizational charts, extracting hierarchical structures, and visualizing knowledge flows.

![Example Visualization](example_visualization.png) *(Example output visualization)*

## Features

- 🖼️ **Computer Vision Detection**: Precise box/position detection using OpenCV
- ✍️ **Text Recognition**: Accurate text extraction using LLaVA
- ↔️ **Relationship Analysis**: Identifies reporting structures and connections
- 📊 **Visualization**: Generates annotated visualizations of the org chart
- 💾 **Structured Output**: Exports analysis as JSON for further processing

## Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU (recommended) with CUDA 11.7+
- PyTorch with CUDA support

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/org-chart-analyzer.git
cd org-chart-analyzer
```

### Usage 

# Ensure CUDA is available (recomended)
python -c "import torch; print(torch.cuda.is_available())"  # Should print True

# Run normally (will auto-detect GPU)
python org_chart_analyzer.py chart.png


## Advanced parameters 
```
python org_chart_analyzer.py company_chart.jpg \
    --output_json company_hierarchy.json \
    --output_viz analysis_result.png \
    --threshold 200 \
    --min_box_area 1000
```

## Sample output 
```
{
  "nodes": [
    {
      "id": 1,
      "text": "CEO",
      "x": 100,
      "y": 50,
      "width": 200,
      "height": 80,
      "center_x": 110,
      "center_y": 60
    }
  ]
}
```
