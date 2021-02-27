#!/bin/bash

# upload to test server
if [ ! -z $PYPI_TEST ] || [ ! -z $PYPI ]; then
  export TWINE_PASSWORD=$TWINE_PASSWORD_PYPITEST
  if [ "$1" = "sdist" ]; then
    python3 -m twine upload -r testpypi --skip-existing dist/*.tar.gz
  fi
  python3 -m twine upload -r testpypi --skip-existing wheelhouse/*.whl
fi

# upload to real pypi server
if [ ! -z $PYPI ]; then
  export TWINE_PASSWORD=$TWINE_PASSWORD_PYPI
  if [ "$1" = "sdist" ]; then
    python3 -m twine upload --skip-existing dist/*.tar.gz
  fi
  python3 -m twine upload --skip-existing wheelhouse/*.whl
fi