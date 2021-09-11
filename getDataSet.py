from icrawler.builtin import BingImageCrawler

cubeTypes = ['2x2 cube', '3x3 cube', '4x4 cube']
cubesNo = 300

filters = dict(
    license='creativecommons'
)

for c in cubeTypes:
    bingCrawler = BingImageCrawler(storage={'root_dir': f'tst/{c}'})
    bingCrawler.crawl(keyword=c, filters=filters, max_num=cubesNo, offset=0)
