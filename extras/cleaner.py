import os
import glob

files1 = glob.glob('output/*')
files2 = glob.glob('data/custom/google_search_images/*')
files = files1 + files2
for f in files:
    os.remove(f)