"""Pin the actual SDK OpenCV binary that exhibited the measured RNG behaviour."""
from pathlib import Path
import hashlib
import cv2
EXPECTED_VERSION='5.0.0'
EXPECTED_BUILD_SHA256='ce2f2890911cf9cf8d7c54cfbc8ccd4fb073cc2b53be1f4c7cbae8e5fd294e1c'
EXPECTED_BINARIES={'cv2.pyd':'eefedf6ee006a84bbc7d3df5ec66046429b8c95dbfd5f8f8f98c5772b91d6d48',
    'opencv_videoio_ffmpeg500_64.dll':'7aabff029d1fa47cf56ede4dbbe05dc62da3208f07cd16471d4c966a33f9aa31'}

def verify():
    root=Path(cv2.__file__).parent
    assert cv2.__version__==EXPECTED_VERSION
    assert hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest()==EXPECTED_BUILD_SHA256
    for name,value in EXPECTED_BINARIES.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==value,(name,'runtime changed')
    return dict(opencv_version=EXPECTED_VERSION,build_information_sha256=EXPECTED_BUILD_SHA256,binary_hashes=EXPECTED_BINARIES)
