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
import time
from tqdm import tqdm


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
    model_name = get_model_name_from_path(model_cfgs['model_path'])
    tokenizer, model, _, _ = load_pretrained_model(
            model_cfgs['model_path'], 
            model_cfgs['model_base'], 
            model_name, 
            cache_dir=model_cfgs['cache_dir'],
            load_8bit=False, 
            load_4bit=False,
            device="cuda")
    doc_image_processor = DocProcessor(
        image_size=448, 
        anchors=model_cfgs['anchors'], 
        add_global_img=model_cfgs['add_global_img'], 
        add_textual_crop_indicator=model_cfgs['add_textual_crop_indicator'])

    results = {}

    for d in tqdm(data):
        images = [osp.join(args.image_folder, i) for i in d['image']]
        images = [i for i in images if i not in results]

        if len(images) == 0:
            continue

        query = d['messages'][0]['content'] + ' Use 1 to 3 words to answer.'
        image_tensor, patch_positions, text = doc_image_processor(images=images, query=query)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
        with torch.inference_mode():
            image_tokens = model.encode_images(image_tensor, patch_positions)
        image_tokens_count = image_tokens.reshape(-1, image_tokens.shape[-1]).shape[0] # bug potentially, only support 1 image

        results.update({
            images[0]: image_tokens_count
        })

    average_count = int(sum(results.values()) / len(results))
    results.update({
        'average_count': average_count
    })
    print(f"Average token count: {average_count}")

    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)

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
    