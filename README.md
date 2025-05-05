Ovis 2 is a Vision Language Model that performs OCR particularly well.

The purpose of this project is to generate training data to finetune Ovis 2 for converting pdfs and
screenshots into a text format ideal for chunking into a RAG system. The ideal format:

1) Maintains the main flow of the document regardless of formatting
2) Discards formatting and text that is an artifact of the medium (such as navigation links
and page numbers)
3) Preserves the heirarchy of headings and sub headings
4) Seperates out but preserves footnotes, figures, and citations

We're using a large Ovis 2 model to help with creating data that's easy to polish. Ultimately 
we would fine tune (via LORA) the 1B parameter model.

To install:

    pip install -U pip wheel
    pip install -r requirements.txt
    pip install flash-attn --no-build-isolation

Feedback and collaboration encouraged
