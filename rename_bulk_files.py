from glob import glob
import os


prefix = "20_"

images = os.listdir("all_powers")
print(images)

os.chdir("all_powers")

[os.rename(f, "{}{}".format(prefix, f)) for f in images]
