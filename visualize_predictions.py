"""
Visualize InternVL3 Pointing Model Predictions
Single image inference with visualization
"""

import os
import re
import argparse

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForImageTextToText


class PointingVisualizer:
    """Visualize pointing predictions on a single image"""

    def __init__(self, model_path: str, device: str = "cuda"):
        """Initialize the visualizer with model"""
        self.device = device
        self.model_path = model_path

        print(f"Loading model from: {model_path}")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        print(f"Model loaded successfully on {device}")

    @staticmethod
    def parse_coordinates(text: str):
        """Extract (x, y) coordinates from model output"""
        match = re.search(r'\((\d+\.?\d*),\s*(\d+\.?\d*)\)', text)
        if match:
            return float(match.group(1)), float(match.group(2))
        return None, None

    def predict_and_visualize(self, image_path: str, question: str,
                             output_path: str = None, max_new_tokens: int = 50):
        """Generate prediction and visualize on image"""

        # Load the image
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        # Prepare the message
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
            ]
        ]

        # Process input - separate text and images
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device, dtype=torch.bfloat16)

        # Generate prediction
        print("Generating prediction...")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        prediction = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0]

        print(f"Prediction: {prediction}")

        # Parse coordinates
        pred_x, pred_y = self.parse_coordinates(prediction)

        if pred_x is None or pred_y is None:
            print(f"Error: Could not parse coordinates from: {prediction}")
            return None

        # Convert to pixel coordinates
        pixel_x = int(pred_x * width)
        pixel_y = int(pred_y * height)

        print(f"Normalized coords: ({pred_x:.3f}, {pred_y:.3f})")
        print(f"Pixel coords: ({pixel_x}, {pixel_y})")

        # Draw on image
        draw = ImageDraw.Draw(image)

        # Draw red crosshair
        marker_size = 20
        line_width = 3

        draw.line(
            [(pixel_x - marker_size, pixel_y), (pixel_x + marker_size, pixel_y)],
            fill='red',
            width=line_width
        )
        draw.line(
            [(pixel_x, pixel_y - marker_size), (pixel_x, pixel_y + marker_size)],
            fill='red',
            width=line_width
        )

        # Draw circle
        circle_radius = 15
        draw.ellipse(
            [
                (pixel_x - circle_radius, pixel_y - circle_radius),
                (pixel_x + circle_radius, pixel_y + circle_radius)
            ],
            outline='red',
            width=line_width
        )

        # Add text overlay
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()

        text_lines = [
            f"Q: {question[:80]}...",
            f"Prediction: {prediction}",
            f"Coords: ({pred_x:.3f}, {pred_y:.3f})",
            f"Pixels: ({pixel_x}, {pixel_y})"
        ]

        text_y = 10
        for line in text_lines:
            bbox = draw.textbbox((10, text_y), line, font=font)
            draw.rectangle(bbox, fill='black')
            draw.text((10, text_y), line, fill='white', font=font)
            text_y += 25

        # Save image
        if output_path is None:
            output_path = "output_prediction.jpg"

        image.save(output_path)
        print(f"\nSaved visualization to: {output_path}")

        return {
            'prediction': prediction,
            'normalized_coords': (pred_x, pred_y),
            'pixel_coords': (pixel_x, pixel_y),
            'output_image': output_path
        }


def main():
    parser = argparse.ArgumentParser(
        description="Visualize InternVL3 Pointing Model Prediction"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to finetuned model checkpoint"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question/prompt for the model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_prediction.jpg",
        help="Path to save output image"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on"
    )

    args = parser.parse_args()

    # Initialize visualizer
    visualizer = PointingVisualizer(
        model_path=args.model_path,
        device=args.device
    )

    # Generate and visualize prediction
    visualizer.predict_and_visualize(
        image_path=args.image,
        question=args.question,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
