"""
Batch Prediction Script for InternVL3 Pointing Model
Runs predictions for all images and all test questions
  python batch_predict.py \
      --model_path output/internvl3_pointing \
      --questions test_questions.json \
      --images_dir /home/yu/Downloads/finetune_data/LLaMA-Factory \
      --output_dir batch_predictions

"""

import os
import json
import glob
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText


class BatchPointingPredictor:
    """Batch prediction for pointing tasks"""

    def __init__(self, model_path: str, device: str = "cuda"):
        """Initialize with model"""
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
        print(f"Model loaded successfully on {device}\n")

    @staticmethod
    def parse_coordinates(text: str):
        """Extract (x, y) coordinates from model output"""
        match = re.search(r'\((\d+\.?\d*),\s*(\d+\.?\d*)\)', text)
        if match:
            return float(match.group(1)), float(match.group(2))
        return None, None

    def predict_single(self, image_path: str, question: str, max_new_tokens: int = 50):
        """Generate prediction for a single image-question pair"""
        image = Image.open(image_path).convert('RGB')

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            [messages],
            images=[image],
            padding=True,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        prediction = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0]

        return prediction

    def draw_prediction(self, image_path: str, pred_x: float, pred_y: float,
                       question: str, prediction_text: str, object_name: str,
                       output_path: str):
        """Draw predicted point on image and save"""
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        pixel_x = int(pred_x * width)
        pixel_y = int(pred_y * height)

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
            f"Object: {object_name}",
            f"Prediction: {prediction_text}",
            f"Coords: ({pred_x:.3f}, {pred_y:.3f})",
            f"Pixels: ({pixel_x}, {pixel_y})"
        ]

        text_y = 10
        for line in text_lines:
            bbox = draw.textbbox((10, text_y), line, font=font)
            draw.rectangle(bbox, fill='black')
            draw.text((10, text_y), line, fill='white', font=font)
            text_y += 25

        image.save(output_path)

    def batch_predict(self, questions_json: str, images_dir: str,
                     output_dir: str = "batch_predictions"):
        """Run predictions on all images and questions"""

        # Load questions
        print(f"Loading questions from: {questions_json}")
        with open(questions_json, 'r') as f:
            questions_data = json.load(f)

        # Find all JPG images
        image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

        if not image_files:
            print(f"Error: No .jpg files found in {images_dir}")
            return

        print(f"Found {len(image_files)} images:")
        for img in image_files:
            print(f"  - {os.path.basename(img)}")

        print(f"\nFound {len(questions_data)} test questions")
        print(f"Total predictions to generate: {len(image_files)} × {len(questions_data)} = {len(image_files) * len(questions_data)}\n")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Store all results
        all_results = []
        success_count = 0
        failed_count = 0

        # Process all combinations
        total = len(image_files) * len(questions_data)
        pbar = tqdm(total=total, desc="Processing")

        for img_idx, image_path in enumerate(image_files):
            image_name = Path(image_path).stem

            for q_idx, question_data in enumerate(questions_data):
                try:
                    # Extract question text
                    question = question_data['conversations'][0]['value'].replace('<image>', '').strip()
                    object_name = question_data['object']

                    # Generate prediction
                    prediction = self.predict_single(image_path, question)

                    # Parse coordinates
                    pred_x, pred_y = self.parse_coordinates(prediction)

                    if pred_x is None or pred_y is None:
                        failed_count += 1
                        all_results.append({
                            'image': os.path.basename(image_path),
                            'object': object_name,
                            'question': question,
                            'prediction': prediction,
                            'status': 'parsing_failed'
                        })
                        pbar.update(1)
                        continue

                    # Create output filename
                    output_filename = f"{image_name}_{object_name.replace(' ', '_')}_q{q_idx}.jpg"
                    output_path = os.path.join(output_dir, output_filename)

                    # Draw and save visualization
                    self.draw_prediction(
                        image_path=image_path,
                        pred_x=pred_x,
                        pred_y=pred_y,
                        question=question,
                        prediction_text=prediction,
                        object_name=object_name,
                        output_path=output_path
                    )

                    success_count += 1

                    # Store result
                    all_results.append({
                        'image': os.path.basename(image_path),
                        'object': object_name,
                        'question': question,
                        'prediction': prediction,
                        'normalized_coords': {'x': pred_x, 'y': pred_y},
                        'output_image': output_filename,
                        'status': 'success'
                    })

                except Exception as e:
                    failed_count += 1
                    print(f"\nError processing {image_path} with {object_name}: {e}")
                    all_results.append({
                        'image': os.path.basename(image_path),
                        'object': object_name,
                        'status': 'error',
                        'error': str(e)
                    })

                pbar.update(1)

        pbar.close()

        # Save results JSON
        results_file = os.path.join(output_dir, 'all_predictions.json')
        with open(results_file, 'w') as f:
            json.dump({
                'model_path': self.model_path,
                'questions_file': questions_json,
                'images_directory': images_dir,
                'total_images': len(image_files),
                'total_questions': len(questions_data),
                'total_predictions': len(image_files) * len(questions_data),
                'successful': success_count,
                'failed': failed_count,
                'results': all_results
            }, f, indent=2)

        # Print summary
        print("\n" + "="*70)
        print("BATCH PREDICTION COMPLETE")
        print("="*70)
        print(f"Images processed:        {len(image_files)}")
        print(f"Questions per image:     {len(questions_data)}")
        print(f"Total predictions:       {len(image_files) * len(questions_data)}")
        print(f"Successful:              {success_count}")
        print(f"Failed:                  {failed_count}")
        print(f"\nOutput directory:        {output_dir}")
        print(f"Results JSON:            {results_file}")
        print("="*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch Prediction for InternVL3 Pointing Model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to finetuned model checkpoint"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="test_questions.json",
        help="Path to test questions JSON file"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=".",
        help="Directory containing .jpg images"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="batch_predictions",
        help="Directory to save predictions"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on"
    )

    args = parser.parse_args()

    # Initialize predictor
    predictor = BatchPointingPredictor(
        model_path=args.model_path,
        device=args.device
    )

    # Run batch prediction
    predictor.batch_predict(
        questions_json=args.questions,
        images_dir=args.images_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
