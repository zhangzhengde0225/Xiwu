
import requests
import re

def parse_arxiv_paper_page(_url_):
    """
    解析arxiv的论文页面，获取论文的标题、作者、年份、摘要、类别、doi、comment
    """
    import os
    from bs4 import BeautifulSoup
    print('正在获取文献名！')
    print(_url_)

    # arxiv_recall = {}
    # if os.path.exists('./arxiv_recall.pkl'):
    #     with open('./arxiv_recall.pkl', 'rb') as f:
    #         arxiv_recall = pickle.load(f)

    # if _url_ in arxiv_recall:
    #     print('在缓存中')
    #     return arxiv_recall[_url_]

    # proxies, = get_conf('proxies')
    proxies = {}
    res = requests.get(_url_, proxies=proxies)

    bs = BeautifulSoup(res.text, 'html.parser')
    other_details = {}

    # get year
    try:
        year = bs.find_all(class_='dateline')[0].text
        year = re.search(r'(\d{4})', year, re.M | re.I).group(1)
        other_details['year'] = year
        abstract = bs.find_all(class_='abstract mathjax')[0].text
        other_details['abstract'] = abstract
    except:
        other_details['year'] = None
        print('年份获取失败')

    # get author
    try:
        authors = bs.find_all(class_='authors')[0].text
        authors = authors.split('Authors:')[1]
        other_details['authors'] = authors
    except:
        other_details['authors'] = None
        print('authors获取失败')

    # get comment
    try:
        comment = bs.find_all(class_='metatable')[0].text
        real_comment = None
        for item in comment.replace('\n', ' ').split('   '):
            if 'Comments' in item:
                real_comment = item
        if real_comment is not None:
            other_details['comment'] = real_comment
        else:
            other_details['comment'] = None
    except:
        other_details['comment'] = None
        print('年份获取失败')

    # 获取类别
    try:
        subjects = bs.find_all(class_='primary-subject')[0].text
        other_details['subjects'] = subjects
    except:
        other_details['subjects'] = None
        print('subjects获取失败')

    # 获取doi
    try:
        # doi = bs.find_all(class_='full-text')[0].text
        doi = bs.find('meta', {'name': 'citation_doi'})['content']
        other_details['doi'] = doi
    except:
        other_details['doi'] = None
        print('doi获取失败')
    
    # 获取标题
    try:
        title = bs.find('meta', {'name': 'citation_title'})['content']
        other_details['title'] = title
    except:
        other_details['title'] = None
        print('title获取失败')

    title_str = BeautifulSoup(
        res.text, 'html.parser').find('title').contents[0]
    print('获取成功：', title_str)
    # arxiv_recall[_url_] = (title_str+'.pdf', other_details)
    # with open('./arxiv_recall.pkl', 'wb') as f:
    #     pickle.dump(arxiv_recall, f)

    return title_str+'.pdf', other_details


def parse_arxiv_format_page(_url_, proxies={}):
    from bs4 import BeautifulSoup
    res = requests.get(_url_, proxies=proxies)
    soup = BeautifulSoup(res.text, 'html.parser')
    

    # get_source
    try:
        source = soup.find_all(class_='source')[0].text
        source = source.split('Source:')[1]
    except:
        source = None
        print('source获取失败')
    pass