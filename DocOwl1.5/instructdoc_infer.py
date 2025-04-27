import torch
from PIL import Image
from transformers import TextStreamer
import os, os.path as osp
import argparse
import json

from mplug_docowl.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_docowl.conversation import conv_templates, SeparatorStyle
from mplug_docowl.model.builder import load_pretrained_model
from mplug_docowl.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from mplug_docowl.processor import DocProcessor
from icecream import ic
import time
from tqdm import tqdm


class DocOwlInfer():
    def __init__(self, 
                 model_path,
                 model_base=None, 
                 cache_dir='./checkpoints',
                 context_len=3600,
                 anchors='grid_9', 
                 add_global_img=True,
                 add_textual_crop_indicator=True, 
                 load_8bit=False, 
                 load_4bit=False, 
                 temperature=1.0, 
                 max_new_tokens=512):
        model_name = get_model_name_from_path(model_path)
        # ic(model_name)

        self.tokenizer, self.model, _, _ = load_pretrained_model(
            model_path, 
            model_base, 
            model_name, 
            cache_dir=cache_dir,
            load_8bit=load_8bit, 
            load_4bit=load_4bit,
            device="cuda")
        self.doc_image_processor = DocProcessor(image_size=448, anchors=anchors, add_global_img=add_global_img, add_textual_crop_indicator=add_textual_crop_indicator)
        # self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.context_len = context_len
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def inference(self, images, query):
        if '<|image|>' not in query:
            query = '<|image|>'*len(images)+query

        image_tensor, patch_positions, text = self.doc_image_processor(images=images, query=query)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        patch_positions = patch_positions.to(self.model.device)

        # ic(image_tensor.shape, patch_positions.shape, text)

        conv = conv_templates["mplug_owl2"].copy()
        roles = conv.roles # ("USER", "ASSISTANT")

        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # ic(prompt)

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        
        # ic(input_ids)
        # truncate input_ids to context_len (this will limit the number of text tokens generated)
        if input_ids.shape[1] > self.context_len:
            input_ids = input_ids[:, :self.context_len]

        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                patch_positions=patch_positions,
                do_sample=False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                # streamer=self.streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

        return outputs.replace('</s>', '')

def main(args):
    assert osp.exists(args.test_json), f"Test json not found: {args.test_json}"
    assert not osp.exists(args.output_json), f"Output json already exists: {args.output_json}"

    # Load source json
    with open(args.test_json, "r") as f:
        data = json.load(f)

    model_cfgs = dict(
        model_path=args.model_path,
        cache_dir=args.cache_dir,
        model_base=args.model_base,
        context_len=args.context_len,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        anchors=args.anchors,
        add_global_img=args.add_global_img,
        add_textual_crop_indicator=args.add_textual_crop_indicator
    )

    print(model_cfgs)

    docowl = DocOwlInfer(
        model_cfgs['model_path'], 
        model_cfgs['model_base'],
        cache_dir=model_cfgs['cache_dir'], 
        context_len=model_cfgs['context_len'],
        anchors=model_cfgs['anchors'], 
        add_global_img=model_cfgs['add_global_img'],
        add_textual_crop_indicator=model_cfgs['add_textual_crop_indicator'],
        max_new_tokens=model_cfgs['max_new_tokens'],
        temperature=model_cfgs['temperature'])

    results = []

    for d in tqdm(data):
        images = [osp.join(args.image_folder, i) for i in d['image']]
        
        query = d['messages'][0]['content'] + ' Use 1 to 3 words to answer.'
        answer = docowl.inference(images, query)

        d['messages'][1]['content'] = answer
        results.append(d)

    with open(args.output_json, "w") as f:
        json.dump(results, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("test_json", type=str)
    parser.add_argument("output_json", type=str)
    parser.add_argument("--model_path", type=str, default='mPLUG/DocOwl1.5')
    parser.add_argument("--cache_dir", type=str, default='./checkpoints')
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--context_len", type=int, default=9600)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--anchors", type=str, default='grid_9')
    parser.add_argument("--add_global_img", action='store_true')
    parser.add_argument("--add_textual_crop_indicator", action='store_true')
    parser.add_argument("--image_folder", type=str, default='')

    args = parser.parse_args()
    main(args)
    