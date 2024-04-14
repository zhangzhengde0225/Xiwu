"""
下载arxiv论文
"""
import os, sys
import requests
from pathlib import Path
import json
import re
import damei as dm

from utils import parse_arxiv_paper_page, parse_arxiv_format_page
here = Path(__file__).parent.absolute()

logger = dm.get_logger('download_arxiv_paper')



class arXiv(object):

    def __init__(self, dataset_dir=None) -> None:
        dataset_dir = dataset_dir if dataset_dir is not None else f'{here}/arxiv_papers'
        self.papers_file = f'{dataset_dir}/arxiv_papers.json'
        self.save_dir = f'{dataset_dir}/files'
        self.papers_info = self._load_papers_info()
    
    def _load_papers_info(self):
        if os.path.exists(self.papers_file):
            with open(self.papers_file, 'r') as f:
                papers_info = json.load(f)
        else:
            os.makedirs(f'{here}/arxiv_papers', exist_ok=True)
            papers_info = {}
            self._save_papers_info(papers_info)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        return papers_info
    
    def _save_papers_info(self, data=None):
        data = data if data is not None else self.papers_info
        with open(self.papers_file, 'w') as f:
            json.dump(data, f, indent=4)

    def save_download_info_to_json(self, file_name, info, **kwargs):
        info.update(kwargs)
        info['file_name'] = file_name
        rel_save_dir = self.save_dir.replace(f'{here}/', '')
        info['file_path'] = f'{rel_save_dir}/{file_name}'
        
        key = info['doi']
        key = key if key is not None else f'{info["year"]}_{info["title"]}'
        
        self.papers_info[key] = info
        self._save_papers_info()


    def request_pdf_and_save(self, url_pdf, save_path, proxies={}):
        """
        根据pdf链接请求并保存到文件
        :param url_pdf: pdf链接, 如：'https://arxiv.org/pdf/hep-ex/0701001.pdf'
        :param save_path: 保存路径
        """
        try:
            r = requests.get(url_pdf, proxies=proxies)
            with open(save_path, 'wb') as f:
                f.write(r.content)
            return True
        except Exception as e:
            logger.debug(f'下载失败：{url_pdf}， {e}')
            return False
        
    def auto_completion_url(self, url_pdf):
        """
        自动补全arxiv的网页链接
        :param url_pdf: 可以所paper_id如hep-ex/0701001，也可以所pdf链接, 如：'https://arxiv.org/pdf/hep-ex/0701001.pdf'
        """
        if 'arxiv.org' not in url_pdf:
            if ('.' in url_pdf) and ('/' not in url_pdf):
                new_url = 'https://arxiv.org/abs/'+url_pdf
                print('下载编号：', url_pdf, '自动定位：', new_url)
                # download_arxiv_(new_url)
                return self.auto_completion_url(new_url)
            else:
                print('不能识别的URL！')
                return None
        if 'abs' in url_pdf:
            url_pdf = url_pdf.replace('abs', 'pdf')
            url_pdf = url_pdf + '.pdf'

        url_abs = url_pdf.replace('.pdf', '').replace('pdf', 'abs')
        return url_abs

    def download_arxiv_(self, url_pdf, download_dir=None, proxies={}):
        """
        下载论文，根据doi
        """
        download_dir = download_dir if download_dir is not None else self.save_dir
        url_abs = self.auto_completion_url(url_pdf)
        url_pdf = f'{url_abs}.pdf'
        
        title, other_info = parse_arxiv_paper_page(url_abs)
        paper_id = title.split()[0]  # '[1712.00559]'
        subjects = other_info['subjects']  # 如果有类别，保存到该类别的文件夹里
        if len(subjects) > 0:
            download_dir = f'{download_dir}/{subjects}'
            os.makedirs(download_dir, exist_ok=True)

        year = other_info['year'] if other_info['year'] is not None else 'Unknown_year'
        doi = other_info['doi'] if other_info['doi'] is not None else 'Unknown_doi'
        file_stem = f'{year} {paper_id} {doi}'  # 年 [编号] doi
        file_stem = file_stem.replace('/', '_')   # 替换掉/

        # 下载和保存tex
        url_format = url_abs.replace('abs', 'format')
        # self.download_tex(url_format, download_dir, proxies=proxies)

        # 下载和保存pdf
        pdf_file_path = f'{download_dir}/{file_stem}.pdf'
        file_name = self.download_pdf(url_abs, pdf_file_path)
        # 保存信息
        self.save_download_info_to_json(file_name, other_info)
    
    def download_tex(self, url, download_dir=None, proxies={}):
        # ret = parse_arxiv_format_page(url_tex)
        import zipfile
        import io
        e_print_url = url.replace('format', 'e-print')
        url = "https://arxiv.org/e-print/hep-ex/0701001v2"
        # 获得真实下载地址
        response = requests.head(url, proxies=proxies, allow_redirects=True)
        print(response.history)
        print(response.url)


        response = requests.get(url, proxies=proxies)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall('./')

        a = response.content.decode('gbk')
        with open('./aa', 'wb') as f:
            f.write(response.content)
        # cmd = f'wget -O {download_dir} {e_print_url}'
        # print(cmd)
        # os.system(cmd)
        pass

    def download_pdf(self, url_abs, file_path):
        
        print('下载中')
        # proxies, = get_conf('proxies')
        requests_pdf_url = url_abs.replace('abs', 'pdf') + '.pdf'
        ok = self.request_pdf_and_save(requests_pdf_url, file_path)
        if not ok:
            print('下载失败')
            return None
        print('下载完成')
        return file_path

        # print('输出下载命令：','aria2c -o \"%s\" %s'%(title_str,url_pdf))
        # subprocess.call('aria2c --all-proxy=\"172.18.116.150:11084\" -o \"%s\" %s'%(download_dir+title_str,url_pdf), shell=True)

    
    
        

if __name__ == '__main__':
    arxiv = arXiv(dataset_dir='/data/public/arxiv_papers')
    arxiv.download_arxiv_('https://arxiv.org/abs/hep-ex/0701001')

