# coding=utf-8

from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup,find_packages

setup( name         = 'cgml',
       version      = '1.0.1',
       description  = 'Machine Learning with Computational Graphs',
       author       = 'Timo Erkkilä',
       author_email = 'timo.erkkila@gmail.com',
       url          = 'http://github.com/terkkila/cgml/',
       packages     = find_packages(),
       #package_dir  = {'cgml': 'src/cgml'},
       scripts      = ['bin/cgml'])
#       data_files   = [('bin', ['bin/cgml'])])
