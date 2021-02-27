#!/bin/bash

while [ -n "$1" ]; do
    case "$1" in
        -h|--help)
            echo "Usage: ./install_wasserstein.sh [--prefix|-p PREFIX] [--include-path|-i INCLUDE_PATH] [--share-path|-s SHARE_PATH] [--boost|-b] [--help|-h]";
            exit;;
        -p|--prefix)
            PREFIX="$2";
            shift;;
        -i|--include-path)
            WASSERSTEIN_INCLUDE_PATH="$2";
            shift;;
        -s|--share-path)
            WASSERSTEIN_SHARE_PATH="$2"
            shift;;
        -b|--boost)
            BOOST=true;;
        *)
            echo "Unknown option $1";
            exit;;
    esac
    shift
done

# handle install prefix
if [ -z $PREFIX ]; then
    PREFIX="/usr/local"
    read -p "Proceed using default prefix (/usr/local)? [y/n] " ans
    if [ "$ans" != "y" ]; then
        echo "Terminating installation"
        exit
    fi
fi

# handle include path
if [ -z $WASSERSTEIN_INCLUDE_PATH ]; then
    WASSERSTEIN_INCLUDE_PATH="$PREFIX/include"
fi
WASSERSTEIN_INCLUDE_PATH="$WASSERSTEIN_INCLUDE_PATH/wasserstein"

# handle share path
if [ -z $WASSERSTEIN_SHARE_PATH ]; then
    WASSERSTEIN_SHARE_PATH="$PREFIX/share"
fi
WASSERSTEIN_SHARE_PATH="$WASSERSTEIN_SHARE_PATH/wasserstein/swig"

# remove old directories
rm -rfv $WASSERSTEIN_INCLUDE_PATH $WASSERSTEIN_SHARE_PATH

# make directories
mkdir -pv "$WASSERSTEIN_INCLUDE_PATH"
#mkdir -pv "$WASSERSTEIN_INCLUDE_PATH/internal"
mkdir -pv "$WASSERSTEIN_SHARE_PATH"

# copy main headers
WASSERSTEIN_HEADERS="EMD.hh CorrelationDimension.hh"
echo "Installing Wasserstein headers"
for header in $WASSERSTEIN_HEADERS; do
    cp -av "./wasserstein/$header" "$WASSERSTEIN_INCLUDE_PATH"
done

# copy internal headers
echo "Installing Wasserstein internal headers"
cp -aRv ./wasserstein/internal $WASSERSTEIN_INCLUDE_PATH
#WASSERSTEIN_INTERNAL_HEADERS="EMDUtils.hh Event.hh HistogramUtils.hh NetworkSimplex.hh PairwiseDistance.hh"
#for header in $WASSERSTEIN_INTERNAL_HEADERS; do
#    cp -av "./wasserstein/internal/$header" "$WASSERSTEIN_INCLUDE_PATH/internal"
#done

# copy swig files
WASSERSTEIN_SWIG_INTERFACES="wasserstein.i wasserstein_common.i"
echo "Installing Wasserstein SWIG interfaces"
for header in $WASSERSTEIN_SWIG_INTERFACES; do
    cp -av "./wasserstein/swig/$header" "$WASSERSTEIN_SHARE_PATH"
done

# determine if we're installing boost
if [ ! -z "$BOOST" ] && [ "$BOOST" != "false" ]; then
    BOOST_INCLUDE_PATH="$WASSERSTEIN_INCLUDE_PATH/.."
    echo "Installing boost histogram package to: $BOOST_INCLUDE_PATH"
    cp -aR "./boost" "$BOOST_INCLUDE_PATH"
fi
