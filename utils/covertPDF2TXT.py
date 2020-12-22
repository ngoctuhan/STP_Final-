from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import io
# import doctext


def pdf2txt(inPDFfile, outTXTFile, save=False):

    inFile = open(inPDFfile, 'rb')
    resMgr = PDFResourceManager()
    retData = io.StringIO()
    TxtConverter = TextConverter(resMgr, retData, laparams=LAParams())
    interpreter = PDFPageInterpreter(resMgr, TxtConverter)

    for page in PDFPage.get_pages(inFile):
        interpreter.process_page(page)

    txt = retData.getvalue()
    if save:
        with open(outTXTFile, 'w') as f:
            f.write(txt)
    return txt


def docx2txt(docfile, outputF, save=False):
    # or you may directly enter the path to docs file. >>>doc_text = doctext.DocFile(doc=path_to_docx_file)
    # doc_text = doctext.DocFile(url=docfile)
    # text = doc_text.get_text()
    # if save:
    #     with open(outputF, 'w') as f:
    #         f.write(text)
    # return text
    pass
