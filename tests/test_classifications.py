import time

from shelterbelts.classifications.lidar import lidar


def test_basic():
    """Quickly test all the functions"""
    lidar('data/Young201709-LID1-C3-AHD_6306194_55_0002_0002.laz', outdir='outdir', stub='g2_26729', resolution=1, category5=True)


def test_lidar():
    """More comprehensive lidar tests: 2 resolutions, 2 heights, with or without category_5, binary or percent cover"""
    sample_laz = 'data/Young201709-LID1-C3-AHD_6306194_55_0002_0002.laz'  # Downloaded this from ELVIS

    lidar(sample_laz, outdir='outdir', stub='g2_26729', resolution=1, category5=True)
    lidar(sample_laz, outdir='outdir', stub='g2_26729', resolution=10, category5=True)
    lidar(sample_laz, outdir='outdir', stub='g2_26729', resolution=1, height_threshold=2, category5=False)
    lidar(sample_laz, outdir='outdir', stub='g2_26729', resolution=1, height_threshold=5, category5=False)
    lidar(sample_laz, outdir='outdir', stub='g2_26729', resolution=10, height_threshold=2, category5=False)
    lidar(sample_laz, outdir='outdir', stub='g2_26729', resolution=10, category5=True, binary=False)
    lidar(sample_laz, outdir='outdir', stub='g2_26729', resolution=10, height_threshold=2, category5=False, binary=False)


if __name__ == '__main__':
    print("testing classifications")
    start = time.time()

    test_lidar()

    print(f"tests successfully completed in {time.time() - start} seconds")