import cv2
import numpy as np
from PIL import Image
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
import argparse
import json
import matplotlib.pyplot as plt

class OrgChartAnalyzer:
    def __init__(self):
        # Initialize OpenCV parameters
        self.cv_params = {
            'threshold': 180,
            'min_box_area': 500,
            'max_box_area': 50000,
            'contour_approximation': cv2.CHAIN_APPROX_SIMPLE
        }
        
        # Initialize LLaVA
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llava_model = None
        self.processor = None
        self._initialize_llava()

    def _initialize_llava(self):
        """Initialize LLaVA model for text recognition"""
        try:
            model_id = "llava-hf/llava-1.5-7b-hf"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=torch.float16
            ).to(self.device)
        except Exception as e:
            print(f"Error initializing LLaVA: {e}")
            raise

    def detect_boxes(self, image_path):
        """
        Detect boxes in organizational chart using OpenCV
        Returns list of bounding boxes in format (x, y, w, h)
        """
        # Load image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Thresholding
        _, thresh = cv2.threshold(
            gray, 
            self.cv_params['threshold'], 
            255, 
            cv2.THRESH_BINARY_INV
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, 
            cv2.RETR_TREE, 
            self.cv_params['contour_approximation']
        )
        
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # Filter by size
            if (self.cv_params['min_box_area'] < area < self.cv_params['max_box_area'] 
                and w > 30 and h > 30):
                boxes.append((x, y, w, h))
                
        return boxes, img.shape

    def recognize_text(self, image, boxes):
        """
        Use LLaVA to recognize text in each detected box
        """
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        results = []
        
        for i, (x, y, w, h) in enumerate(boxes):
            # Crop the box from image
            box_img = pil_image.crop((x, y, x+w, y+h))
            
            # Prepare prompt
            prompt = "What text is in this image? Return ONLY the text content with no additional commentary."
            
            # Process with LLaVA
            inputs = self.processor(
                text=prompt, 
                images=box_img, 
                return_tensors="pt"
            ).to(self.device, torch.float16)
            
            output = self.llava_model.generate(**inputs, max_new_tokens=50)
            text = self.processor.decode(output[0], skip_special_tokens=True)
            
            # Clean up text
            text = text.replace(prompt, "").strip()
            
            results.append({
                "id": i+1,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "text": text,
                "center_x": x + w//2,
                "center_y": y + h//2
            })
            
        return results

    def analyze_relationships(self, image, nodes):
        """
        Use LLaVA to analyze hierarchical relationships between nodes
        """
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Visualize nodes on image for LLaVA
        marked_img = image.copy()
        for node in nodes:
            cv2.rectangle(marked_img, 
                         (node['x'], node['y']), 
                         (node['x']+node['width'], node['y']+node['height']), 
                         (0, 255, 0), 2)
            cv2.putText(marked_img, str(node['id']), 
                       (node['x'], node['y']-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Convert to PIL for LLaVA
        marked_pil = Image.fromarray(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB))
        
        # Create relationship analysis prompt
        node_list = "\n".join([f"{n['id']}: {n['text']}" for n in nodes])
        prompt = f"""Analyze this organizational chart with numbered nodes:
{node_list}

Identify reporting relationships between these nodes. 
Return ONLY a JSON list of connections where each item has:
- "source": parent node ID
- "target": child node ID
Example output:
[{{"source": 1, "target": 2}}, {{"source": 1, "target": 3}}]"""
        
        # Process with LLaVA
        inputs = self.processor(
            text=prompt, 
            images=marked_pil, 
            return_tensors="pt"
        ).to(self.device, torch.float16)
        
        output = self.llava_model.generate(**inputs, max_new_tokens=300)
        result = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Extract JSON from result
        try:
            json_start = result.find('[')
            json_end = result.rfind(']') + 1
            relationships = json.loads(result[json_start:json_end])
            return relationships
        except json.JSONDecodeError:
            print("Failed to parse relationships JSON")
            return []

    def visualize_results(self, image, nodes, relationships, output_path=None):
        """Visualize the detected structure"""
        plt.figure(figsize=(12, 8))
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        
        # Draw nodes
        for node in nodes:
            rect = plt.Rectangle(
                (node['x'], node['y']),
                node['width'],
                node['height'],
                fill=False,
                edgecolor='green',
                linewidth=2
            )
            plt.gca().add_patch(rect)
            plt.text(
                node['x'] + node['width']/2,
                node['y'] - 10,
                f"{node['id']}: {node['text']}",
                ha='center',
                va='bottom',
                color='red',
                fontsize=8
            )
        
        # Draw relationships
        for rel in relationships:
            source = next(n for n in nodes if n['id'] == rel['source'])
            target = next(n for n in nodes if n['id'] == rel['target'])
            
            plt.plot(
                [source['center_x'], target['center_x']],
                [source['center_y'], target['center_y']],
                'r--',
                linewidth=1.5,
                alpha=0.7
            )
            plt.scatter(
                [source['center_x'], target['center_x']],
                [source['center_y'], target['center_y']],
                color='blue',
                s=50
            )
        
        plt.axis('off')
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.show()

    def analyze(self, image_path, output_json_path, output_viz_path=None):
        """Complete analysis pipeline"""
        # Step 1: Detect boxes with OpenCV
        boxes, img_shape = self.detect_boxes(image_path)
        image = cv2.imread(image_path)
        
        # Step 2: Recognize text with LLaVA
        nodes = self.recognize_text(image, boxes)
        
        # Step 3: Analyze relationships with LLaVA
        relationships = self.analyze_relationships(image, nodes)
        
        # Prepare output
        result = {
            "nodes": nodes,
            "edges": relationships,
            "image_dimensions": {
                "width": img_shape[1],
                "height": img_shape[0]
            }
        }
        
        # Save results
        with open(output_json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Visualize
        self.visualize_results(image, nodes, relationships, output_viz_path)
        
        return result

def main():
    parser = argparse.ArgumentParser(description="Analyze organizational chart")
    parser.add_argument("image_path", help="Path to organizational chart image")
    parser.add_argument("--output_json", default="org_structure.json", 
                       help="Output JSON file path")
    parser.add_argument("--output_viz", help="Output visualization image path")
    args = parser.parse_args()
    
    analyzer = OrgChartAnalyzer()
    result = analyzer.analyze(args.image_path, args.output_json, args.output_viz)
    
    print(f"Analysis complete. Results saved to {args.output_json}")
    if args.output_viz:
        print(f"Visualization saved to {args.output_viz}")

if __name__ == "__main__":
    main()
