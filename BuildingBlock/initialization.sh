python setup.py build_ext --inplace
python3 -m pip install -e . --use-pep517
pip install open3d==0.18.0 
python initialization_nltk.py