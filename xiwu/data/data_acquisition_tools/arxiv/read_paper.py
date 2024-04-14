import os, sys
from pathlib import Path
import json

here = Path(__file__).parent.absolute()



class PaperReader(object):

    def __init__(self) -> None:
        
        pass


    def read_paper(self, file_path):
        from bs4 import BeautifulSoup
        fp =  file_path
        if ".tex" in fp:
            with open(fp, 'r', encoding='utf-8', errors='replace') as f:
                file_content = f.read()
        if ".pdf" in fp.lower():
            file_content = self.readPdf(fp)
            file_content = BeautifulSoup(''.join(file_content), features="lxml").body.text.encode('gbk', 'ignore').decode('gbk')
        print(file_content)
        self.save2txt(file_content, f'{here}/xx.txt')
        return file_content
    
    def save2txt(self, content, file_name):
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(content)



    @staticmethod
    def readPdf(pdfPath):
        """
        读取pdf文件，返回文本内容
        """
        import pdfminer
        from pdfminer.pdfparser import PDFParser
        from pdfminer.pdfdocument import PDFDocument
        from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
        from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
        from pdfminer.pdfdevice import PDFDevice
        from pdfminer.layout import LAParams
        from pdfminer.converter import PDFPageAggregator

        fp = open(pdfPath, 'rb')

        # Create a PDF parser object associated with the file object
        parser = PDFParser(fp)

        # Create a PDF document object that stores the document structure.
        # Password for initialization as 2nd parameter
        document = PDFDocument(parser)
        # Check if the document allows text extraction. If not, abort.
        if not document.is_extractable:
            raise PDFTextExtractionNotAllowed

        # Create a PDF resource manager object that stores shared resources.
        rsrcmgr = PDFResourceManager()

        # Create a PDF device object.
        # device = PDFDevice(rsrcmgr)

        # BEGIN LAYOUT ANALYSIS.
        # Set parameters for analysis.
        laparams = LAParams(
            char_margin=10.0,
            line_margin=0.2,
            boxes_flow=0.2,
            all_texts=False,
        )
        # Create a PDF page aggregator object.
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        # Create a PDF interpreter object.
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        # loop over all pages in the document
        outTextList = []
        for page in PDFPage.create_pages(document):
            # read the page into a layout object
            interpreter.process_page(page)
            layout = device.get_result()
            for obj in layout._objs:
                if isinstance(obj, pdfminer.layout.LTTextBoxHorizontal):
                    # print(obj.get_text())
                    outTextList.append(obj.get_text())

        return outTextList
    

if __name__ == '__main__':
    pr = PaperReader()
    data_dir = f'/data/public/arxiv_papers'
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                pr.read_paper(file_path)
                exit()