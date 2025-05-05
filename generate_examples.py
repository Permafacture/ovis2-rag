from screenshot import scrape
from extract_pdf import pdf_to_training_images, load_images, annot_indicator, ocr_images

def gen_training_images():

    scrape('http://www.marx2mao.com/M&E/PI.html', 'source_pdfs/marx.pdf')
    
    pdf_to_training_images('source_pdfs/cancer_book_chapter.pdf', 1, 3)
    pdf_to_training_images('source_pdfs/marx.pdf', 1, 4)

# gen_training_images()
for img_set in ('cancer_book_chapter', 'marx'):
    images = load_images(f'training/ocr_training/{img_set}')
    text = ocr_images(images, prompt_name=3)
    with open(f'training/ocr_training/{img_set}/annot.txt', 'w') as fh:
        fh.write(annot_indicator+'\n\n')
        fh.write(text)
