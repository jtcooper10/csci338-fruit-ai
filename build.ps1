Remove-Item -Recurse docs
sphinx-apidoc -o ./docs .
sphinx-build -b html . docs
