#! /bin/sh

VERSION="${1:-v0.6.1}"

echo "Copying bundle for ${VERSION}..."

mkdir -p ${VERSION}/bundle

cp -v ../onnx-web/gui/out/index.html ${VERSION}/index.html
cp -v ../onnx-web/gui/out/config.json ${VERSION}/config.json
cp -v ../onnx-web/gui/out/bundle/main.js ${VERSION}/bundle/main.js

