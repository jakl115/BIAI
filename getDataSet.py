from icrawler.builtin import BingImageCrawler

cubes = ['rubiks cube']
cubesNo = 500

for c in cubes:
    bingCrawler = BingImageCrawler(storage={'root_dir': f'dataset/p'})
    bingCrawler.crawl(keyword=c, filters=None, max_num=cubesNo, offset=0)

notCubes = ['random stuff']
notCubesNo = 500

for c in notCubes:
    bingCrawler = BingImageCrawler(storage={'root_dir': f'dataset/n'})
    bingCrawler.crawl(keyword=c, filters=None, max_num=notCubesNo, offset=0)