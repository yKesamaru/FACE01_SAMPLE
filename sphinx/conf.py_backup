# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FACE01'
copyright = '2022, yKesamaru'
author = 'yKesamaru'
release = '1.4.09'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# import sys, os
# path_list = ['./example/', './face01lib/']
# for path in path_list:
#     sys.path.append(os.path.abspath(path))


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon'
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
# napoleon_use_ivar = False
# napoleon_use_param = True
# napoleon_use_rtype = True
napoleon_preprocess_types = False
# napoleon_type_aliases = None
napoleon_attr_annotations = True

source_suffix = {
    '.rst': 'restructuredtext'
}


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
