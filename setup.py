# coding=utf-8


from distutils.core import setup

setup( name         = 'cgml',
       version      = '1.0',
       description  = 'Machine Learning with Computational Graphs',
       author       = 'Timo Erkkilä',
       author_email = 'timo.erkkila@gmail.com',
       url          = 'http://github.com/terkkila/cgml/',
       packages     = ['cgml'],
       package_dir  = {'cgml': 'src/cgml'},
       data_files   = [('bin', ['bin/cgml-logreg'])])
