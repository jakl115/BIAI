from icrawler.builtin import BingImageCrawler

classes = ['3X3 Cube', '4X4 Cube', '5X5 Cube']
number = 3

for c in classes:
    bingCrawler = BingImageCrawler(storage={'root_dir': f'p/{c.replace("", "")}'})
    bingCrawler.crawl(keyword=c, filters=None, max_num=number, offset=0)