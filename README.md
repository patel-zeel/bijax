## Description
* Add PYPI_USERNAME and PYPI_PASSWORD to your secrets using GitHub GUI.
* Manually change `requirements.txt` to your needs.
* Run `customize.py` to take care of the rest.
* Each time you push a new release, code is automatically published on PyPI via the workflow. `pip install -U <your_package>` should then install the latest version.
