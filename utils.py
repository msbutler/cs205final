
from PIL import Image
import os

""" compresses all .jpg files in the directory and subdirectory to have a certain basewidth"""
def compress_im(root,basewidth):
    print(f"compress imgs in {root}")
    
    for ob in os.listdir(root):
        file_path = os.path.join(root,ob)
        if os.path.isdir(file_path):
            compress_im(file_path,basewidth)
        elif ob[-3:] == "jpg":
            #print(file_path)
            #print(os.path.getsize(file_path))
            img = Image.open(file_path)
            img.show()
            comp_frac = basewidth/img.size[0]
            assert (comp_frac <=1.)
            hsize = int(img.size[1] * comp_frac)
            img = img.resize((basewidth, hsize), Image.ANTIALIAS)
            img.show()
            img.save(file_path)
            #print(os.path.getsize(file_path))
            #assert 0 ==1
            

#compress_im("Train1",1000)