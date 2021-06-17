#!/usr/bin/env bash

echo "Preparing to upload to PyPI ..."

echo "Installing cibuildwheel, twine, and numpy ..."
python3 -m pip install cibuildwheel twine numpy

echo "Building wheels ..."
python3 -m cibuildwheel --output-dir wheelhouse

# upload to test server
if [ $PYPITEST -gt 0 ]; then
  export TWINE_PASSWORD=$TWINE_PASSWORD_PYPITEST
  if [ "$1" = "sdist" ]; then
    python3 setup.py sdist --formats=gztar
    python3 -m twine upload -r testpypi --skip-existing --verbose dist/*.tar.gz
  fi
  python3 -m twine upload -r testpypi --skip-existing --verbose wheelhouse/*.whl
fi

# upload to real pypi server
if [ $PYPI -gt 0 ]; then
  export TWINE_PASSWORD=$TWINE_PASSWORD_PYPI
  if [ "$1" = "sdist" ]; then
    python3 setup.py sdist --formats=gztar
    python3 -m twine upload --skip-existing --verbose dist/*.tar.gz
  fi
  python3 -m twine upload --skip-existing --verbose wheelhouse/*.whl
fi