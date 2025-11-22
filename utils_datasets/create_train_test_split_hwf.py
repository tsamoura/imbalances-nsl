from os import listdir
from os.path import join
import random
import shutil

source = "/home/experiments/data/HWF/Handwritten_Math_Symbols/train/"
dest = "/home/experiments/data/HWF/Handwritten_Math_Symbols/test/"

for cls in range(13):
    onlyfiles = [f for f in listdir(join(source, str(cls)))]
    size = int(len(onlyfiles) * 0.1)
    test_files = random.sample(onlyfiles, size)
    for file in test_files:
        shutil.move(join(source, str(cls), file), join(dest, str(cls), file))
        
