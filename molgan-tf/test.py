# %%
import os
import subprocess
print(os.getcwd())
os.chdir('m1')
print(os.listdir())
os.system(f"python transformer-cnn.py")

# %%

