from icrawler.builtin import BingImageCrawler

classes = ['human', 'tree', 'cat', 'table', 'car']
number = 100

for c in classes:
    bingCrawler = BingImageCrawler(storage={'root_dir': f'dataset/n'})
    bingCrawler.crawl(keyword=c, filters=None, max_num=number, offset=0)
