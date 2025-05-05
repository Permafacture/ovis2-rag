from pathlib import Path
from PIL import Image
import torch
from pdf2image import convert_from_path, convert_from_bytes
from ocr_model import model, text_tokenizer, visual_tokenizer

base = Path(__file__).parent
training_dir = Path(base, 'training', 'ocr_training')
annot_indicator = "---NOT REVIEWED---"
prompt_dir = Path(base, 'prompts')

def is_annotated(annot_file):
    annot_file = Path(annot_file)
    if not annot_file.exists():
        return False
    first_line = open(annot_file).readline().strip()
    return first_line != annot_indicator

def pdf_to_training_images(pdf_path, first_page, last_page, outdir=training_dir, dpi=150):
    name = Path(pdf_path).stem
    images = convert_from_path(pdf_path, dpi=dpi, first_page=first_page, last_page=last_page)
    out_path = Path(outdir, name)
    out_path.mkdir(exist_ok=True, parents=True)
    annot_name = Path(out_path, "annot.txt")
    assert not is_annotated(annot_name), f"Expected annotated output to not exist for {annot_name}"
    for i, image in enumerate(images, start=first_page):
        fname = Path(out_path, name+f"_{i:03}.png")
        image.save(fname)
        print("wrote", fname)

def load_images(img_path):
    images = []
    for img in Path(img_path).glob('*.png'):
        name = img.parts[-1]
        n = int(name[:-4].split('_')[-1])
        images.append((n, Image.open(img)))
    images.sort(key= lambda x: x[0])
    return [x[1] for x in images]

def get_prompt(name):
    pfile = prompt_dir/f'prompt_{name}.txt'
    return open(pfile).read()


def ocr_images(images, prompt=None, prompt_name=None):
    '''OCR the images. To turn pdf pages into images use `convert_from_path(pdf_path)`

    images is a list of consecutive images.
    '''
    # single-image input
    assert prompt or prompt_name
    assert not (prompt and prompt_name)
    if prompt_name:
        prompt = get_prompt(prompt_name)
    max_partition = 9
    query = '<image>\n'*len(images) + prompt
    
    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=4096,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
        )
        result = []
        prompt, input_ids, pixel_values = model.preprocess_inputs(query, images, max_partition=max_partition)
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
        pixel_values = [pixel_values]
        output_ids = model.generate(input_ids, pixel_values=pixel_values,
                                    attention_mask=attention_mask, **gen_kwargs)[0]
        output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
    return output

if __name__ == "__main__":
    images = load_images('training/ocr_training/')
    result = ocr_images(images)
