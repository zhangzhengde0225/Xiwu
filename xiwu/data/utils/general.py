


def plot_word_frequency(word_freq, **kwargs):
    save_path = kwargs.get('save_path', None)
    show = kwargs.get('show', True)
    width = kwargs.get('width', 800)
    height = kwargs.get('height', 400)
    background_color = kwargs.get('background_color', 'white')
    figure_size = kwargs.get('figure_size', (10, 5))

    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    # # 假设你有一个词频字典
    # word_freq = {'Python': 100, 'programming': 50, 'language': 30, 'data science': 20, 'development': 10}
    # 创建一个词云对象，使用generate_from_frequencies方法
    wordcloud = WordCloud(width=width, height=height, background_color=background_color)
    wordcloud.generate_from_frequencies(word_freq)

    # 显示生成的词云图片
    plt.figure(figsize=figure_size)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # 关闭坐标轴
    # 保存
    plt.savefig(save_path)
    if show:
        plt.show()
    return wordcloud
