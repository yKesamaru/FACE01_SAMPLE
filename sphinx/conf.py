# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FACE01'
copyright = '2022, yKesamaru'
author = 'yKesamaru'

import sys, os
sys.path.append(os.path.abspath("example"))
sys.path.append(os.path.abspath("face01lib"))
sys.path.append(os.path.abspath("."))


sys.path.insert(0, os.path.abspath('..'))

# 'sphinx.ext.viewcode',  # Do not use! The source code becomes completely exposed.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon'
]
# extensions = [
#     'sphinx.ext.napoleon',
#     'sphinx.ext.autodoc',
#     'myst_parser'
# ]

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

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
# テーマ
# https://sphinx-themes.org/#theme-sphinx-rtd-theme

# html_theme = 'python_docs_theme'  # pythonドキュメントと同じはずだけど、よくないなぁ。エラーのせいか？
# html_theme = 'classic'  # 表示が崩れる、よくない
# html_theme = 'scrolls'  # 文字化け、よくない
# html_theme = 'bizstyle'  # Nice!一部表示崩れあり…
html_theme = 'sphinx_rtd_theme'  # Read the Docs
html_static_path = ['_static']

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

source_suffix = {
    '.rst': 'restructuredtext'
}
# source_suffix = {
#     '.rst': 'restructuredtext',
#     '.md': 'markdown',
# }