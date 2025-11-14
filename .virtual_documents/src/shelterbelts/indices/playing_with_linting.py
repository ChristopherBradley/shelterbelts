get_ipython().getoutput("pip install 'jupyterlab-lsp' 'python-lsp-server[all]' # jdfkhidufhfldieifheihfesjkdhfibiuerhweoncnc nc ionecn conwce ")


b = 2


bla = b


hksldds


print


get_ipython().getoutput("pylsp -v")


import glob
import pandas as pd
import math


percent_tifs = glob.glob('/Users/christopherbradley/Documents/PHD/Data/Worldcover_Australia/*')


len(percent_tifs)









chunk_size = 100
df = pd.DataFrame(percent_tifs, columns=["filename"])
for i in range(math.ceil(len(df) / chunk_size)):
    chunk = df[i*chunk_size : (i+1)*chunk_size]
    filename = f"percent_tifs_{i}.csv"
    chunk.to_csv(filename, index=False)
    print(filename)
    


df = pd.read_csv("percent_tifs.csv")

