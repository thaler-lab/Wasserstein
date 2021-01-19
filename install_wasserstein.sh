#!/usr/bin/env bash

# check usage
if [ $# -gt 2 ] || [ "$1" = "-h" ]; then
    echo "Usage: ./install_wasserstein.sh [PREFIX=/usr/local] [INCLUDE_BOOST=false]"
    exit
fi

# handle install path
if [ -z "$1" ]; then
    PREFIX="/usr/local"
    read -p "Proceed using default prefix (/usr/local)? [y/n] " ans
    if [ "$ans" != "y" ]; then
        echo "Terminating installation"
        exit
    fi
else
    PREFIX="$1"
fi

if [[ "$PREFIX" == *"/include"* ]]; then
    WASSERSTEIN_PATH="$PREFIX/wasserstein"
else
    WASSERSTEIN_PATH="$PREFIX/include/wasserstein"
fi
echo "Prefix: $PREFIX"
echo "Install location: $WASSERSTEIN_PATH"

# handle existing directories
if [ -d "$WASSERSTEIN_PATH" ]; then
    read -p "Existing Wasserstein installation found at $WASSERSTEIN_PATH. Would you like to replace it? [y/n] " ans
    if [ "$ans" = "y" ]; then
        rm -rf "$WASSERSTEIN_PATH"
        echo "Removed $WASSERSTEIN_PATH"
    else
        echo "Terminating installation"
        exit
    fi
fi

# make directories
mkdir -pv "$WASSERSTEIN_PATH"
mkdir -pv "$WASSERSTEIN_PATH/internal"

# copy main headers
WASSERSTEIN_HEADERS="EMD.hh CorrelationDimension.hh"
echo "Installing Wasserstein headers"
for header in $WASSERSTEIN_HEADERS; do
    cp -av "./wasserstein/$header" "$WASSERSTEIN_PATH"
done

# copy internal headers
WASSERSTEIN_INTERNAL_HEADERS="EMDUtils.hh Event.hh HistogramUtils.hh NetworkSimplex.hh PairwiseDistance.hh"
echo "Installing Wasserstein internal headers"
for header in $WASSERSTEIN_INTERNAL_HEADERS; do
    cp -av "./wasserstein/internal/$header" "$WASSERSTEIN_PATH/internal"
done

# determine if we're installing boost
#if [ ! -z "$2" ] && [ "$2" != "false" ]; then
#    BOOST_PATH="$WASSERSTEIN_PATH/internal/boost"
#    echo "Installing boost histogram package to: $BOOST_PATH"
#    cp -a "./wasserstein/internal/boost" "$BOOST_PATH"
#fi

echo "Wasserstein is installed at: $WASSERSTEIN_PATH"
