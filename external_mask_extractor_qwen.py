import cv2
import torch
import spacy

import numpy as np
from PIL import Image


# GSegment Anything
from segment_anything import build_sam, SamPredictor

import os
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import ImageDraw


import json
import random
import io
import ast
from io import BytesIO
import xml.etree.ElementTree as ET

class ExternalMaskExtractor():
    def __init__(self, device, debug=False) -> None:
        self.device = device

        self.qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

        # Next, load Segment-Anything
        sam_path = '/home/ubuntu/spjain/Grounded-Instruct-Pix2Pix/SAM/weights/sam_vit_h_4b8939.pth'
        sam = build_sam(checkpoint=sam_path).to(device)
        self.sam_predictor = SamPredictor(sam)
        self.debug = debug

    
     # @title Parsing JSON output
    def parse_json(self, json_output):
        # Parsing out the markdown fencing
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
                json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
                break  # Exit the loop once "```json" is found
        return json_output

    def plot_bounding_boxes(self, im, bounding_boxes):
        """
        Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

        Args:
            img_path: The path to the image file.
            bounding_boxes: A list of bounding boxes containing the name of the object
            and their positions in normalized [y1 x1 y2 x2] format.
        """
        width, height = im.size
        # Parsing out the markdown fencing
        bounding_boxes = self.parse_json(bounding_boxes)

        try:
            json_output = ast.literal_eval(bounding_boxes)
        except Exception as e:
            return None

        if not isinstance(json_output, list):
            json_output = [json_output]

        bboxes = []

        # Load the image
        img = im
        if(self.debug == True):
            # Create a drawing object
            draw = ImageDraw.Draw(img)

            # Define a list of colors
            colors = [
            'red',
            'green',
            'blue',
            'yellow',
            'orange',
            'pink',
            'purple',
            'brown',
            'gray',
            'beige',
            'turquoise',
            'cyan',
            'magenta',
            'lime',
            'navy',
            'maroon',
            'teal',
            'olive',
            'coral',
            'lavender',
            'violet',
            'gold',
            'silver',
            ] 

        # Iterate over the bounding boxes
        for i, bounding_box in enumerate(json_output):

            # Convert normalized coordinates to absolute coordinates
            abs_y1 = int(bounding_box["bbox_2d"][1] / 1000 * height)
            abs_x1 = int(bounding_box["bbox_2d"][0] / 1000 * width)
            abs_y2 = int(bounding_box["bbox_2d"][3] / 1000 * height)
            abs_x2 = int(bounding_box["bbox_2d"][2] / 1000 * width)

            if abs_x1 > abs_x2:
                abs_x1, abs_x2 = abs_x2, abs_x1

            if abs_y1 > abs_y2:
                abs_y1, abs_y2 = abs_y2, abs_y1

            if(self.debug == True):
                # Select a color from the list
                color = colors[i % len(colors)]
                # Draw the bounding box
                draw.rectangle(
                    ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=3
                )

            bboxes.append([abs_x1,abs_y1,abs_x2, abs_y2])
        
        if(self.debug == True):
            # Display the image
            img.show()
        bboxes = torch.tensor(bboxes)
        return bboxes


    

    def get_bounding_boxes(self, image, image_path, prompt):
        prompt = f"You are helping me with image editing. Given this prompt - \"{prompt}\" to edit the image, return the bounding boxes of the areas that need to be edited in JSON fomrat like {{\"bbox_2d\":[x1,y1,x2,y2]}}."
        messages = [
            {
                "role": "user",
                "content": [
                    {   
                        "type": "image_url",
                        # You can set the min_pixels and max_pixels to control the size of the image according to your use case.
                        "image": image_path
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        # Preparation for inference
        text_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        image_inputs, video_inputs = process_vision_info(messages)

        # Inference: Generation of the output
        inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        inputs = inputs.to(self.qwen_model.device)
        generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # mm_data = {}
        # if image_inputs is not None:
        #     mm_data['image'] = image_inputs
        # if video_inputs is not None:
        #     mm_data['video'] = video_inputs

        # inputs =  {
        #     'prompt': text_input,
        #     'multi_modal_data': mm_data,
        # }
        # sampling_params = SamplingParams(
        #         temperature=0,
        #         max_tokens=128,
        #         top_k=-1,
        #         stop_token_ids=[]
        #     )
        
        # outputs = self.qwen_model.generate(inputs, sampling_params = sampling_params)
        # for i, output in enumerate(outputs):
        #     output_text = output.outputs[0].text
        return output_text

    def _sam_predict(self, image, boxes):
        image_orig = np.asarray(image)
        # if no boxes were detected
        if boxes is None or boxes.shape[0] == 0:
            return torch.zeros(image_orig.shape[:2], dtype=torch.float32).to('cuda:0')
        # else, proceed with SAM with bbox
        else:
            self.sam_predictor.set_image(image_orig)
            H, W, _ = image_orig.shape
            # boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes, image_orig.shape[:2]).to('cuda:0')
            masks, _, _ = self.sam_predictor.predict_torch(point_coords=None, point_labels=None,
                                                           boxes=transformed_boxes, multimask_output=False)

            masks_sum = sum(masks)[0]
            masks_sum[masks_sum > 1] = 1
            
            return masks_sum


    @torch.no_grad()
    def get_external_mask(self, image, image_path, prompt, mask_dilation_size=11, verbose=False):
        # Extract all noun-phrases
        bounding_boxes_raw = self.get_bounding_boxes(image,image_path, prompt) #use qwen3vl
        bounding_boxes =self.plot_bounding_boxes(image,bounding_boxes_raw)
        # Extract its mask
        external_mask = self._sam_predict(image, bounding_boxes)
        external_mask = cv2.dilate(external_mask.data.cpu().numpy().astype(np.uint8),
                                   kernel=(np.ones((mask_dilation_size, mask_dilation_size), np.uint8)))
        external_mask = Image.fromarray((255*external_mask).astype(np.uint8))
        if verbose:
            external_mask.show()
            
        return external_mask