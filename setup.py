from distutils.core import setup # Need this to handle modules
import py2exe 

setup(
    options = {'py2exe': {'bundle_files': 1, 'compressed': True}},
    console=['main.py'],
    zipfile = None,
) # Calls setup function to indicate that we're dealing with a single console application