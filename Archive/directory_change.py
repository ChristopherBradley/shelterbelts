
# # # Change directory to this repo - this should work on gadi or locally via python or jupyter.
# No longer need this if you've set up the environment correctly
# import os, sys
# repo_name = "shelterbelts"
# if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
#     repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}")
# elif os.path.basename(os.getcwd()) != repo_name:  # Running in a jupyter notebook 
#     repo_dir = os.path.dirname(os.getcwd())       
# else:                                             # Already running from root of this repo. 
#     repo_dir = os.getcwd()
# src_dir = os.path.join(repo_dir, 'src')
# os.chdir(src_dir)
# sys.path.append(src_dir)
# # print(src_dir)
