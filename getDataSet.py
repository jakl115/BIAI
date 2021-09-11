from icrawler.builtin import BingImageCrawler

cubes = ['2x2 cube', '3x3 cube', '4x4 cube']
cubesNo = 400

for c in cubes:
    bingCrawler = BingImageCrawler(storage={'root_dir': f'tst/{c}'})
    bingCrawler.crawl(keyword=c, filters=None, max_num=cubesNo, offset=0)
