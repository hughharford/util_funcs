from os import listdir
from os.path import isfile, join


# prints only the files in the path given
mypath = "."
onlyfiles = [print(f) for f in listdir(mypath) if isfile(join(mypath, f))]
