import subprocess

subprocess.call("jupyter nbconvert --to script ../notebooks/Readersourcing.ipynb --output-dir='../scripts/'")
subprocess.call("jupyter nbconvert --to script ../notebooks/TrueReview.ipynb --output-dir='../scripts/'")
subprocess.call("jupyter nbconvert --to script ../notebooks/Seeder.ipynb --output-dir='../scripts/'")
subprocess.call("jupyter nbconvert --to script ../notebooks/Experiments.ipynb --output-dir='../scripts/'")
