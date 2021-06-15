#!/usr/bin/env bash

echo "Preparing to upload to PyPI ..."

echo "Installing cibuildwheel and twine ..."
python3 -m pip install twine

echo "Building wheels ..."
python3 -m cibuildwheel --output-dir wheelhouse

# upload to test server
if [ ! -z $PYPI_TEST ] || [ ! -z $PYPI ]; then
  export TWINE_PASSWORD=$TWINE_PASSWORD_PYPITEST
  if [ "$1" = "sdist" ]; then
    python3 setup.py sdist --formats=gztar
    python3 -m twine upload -r testpypi --skip-existing --verbose dist/*.tar.gz
  fi
  python3 -m twine upload -r testpypi --skip-existing --verbose wheelhouse/*.whl
fi

# upload to real pypi server
if [ ! -z $PYPI ]; then
  export TWINE_PASSWORD=$TWINE_PASSWORD_PYPI
  if [ "$1" = "sdist" ]; then
    python3 setup.py sdist --formats=gztar
    python3 -m twine upload --skip-existing --verbose dist/*.tar.gz
  fi
  python3 -m twine upload --skip-existing --verbose wheelhouse/*.whl
fi