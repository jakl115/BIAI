from icrawler.builtin import BingImageCrawler

classes = ['human', 'tree', 'cat']
number = 3

for c in classes:
    bingCrawler = BingImageCrawler(storage={'root_dir': f'n/{c.replace("", "")}'})
    bingCrawler.crawl(keyword=c, filters=None, max_num=number, offset=0)
