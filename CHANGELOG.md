# Changelog

All notable changes to this project will be documented in this file. See [commit-and-tag-version](https://github.com/absolute-version/commit-and-tag-version) for commit guidelines.

## 0.1.0 (2023-01-06)


### Features

* add model and output directories 41d93c9
* add vscode workspace with subdir roots 4088bf4
* **api:** add endpoint to get previous outputs 50221af
* **api:** add endpoint to list models 4cb6ce8
* **api:** add endpoints to list platforms/accelerators and pipeline schedulers c70728d
* **api:** add introspection to index endpoint 8c985e9
* **api:** add option to switch between AMD hardware and CPU software rendering 668e46a
* **api:** add remaining inputs params to response bbd0e93
* **api:** cache pipeline between requests (part of [#5](undefined/undefined/undefined/issues/5)) 82e7fbf
* **api:** cache pipeline when changing scheduler, make txt2img logging more verbose cab13f6
* **build:** add basic CI c6579b7
* **build:** add basic python CI 8d3ca31
* **build:** add bundle to JS build, add API image jobs 6d560af
* **build:** add git multi-push target 917f6ce
* **build:** add nginx-based GUI images 5e9890f
* **build:** add release target ce759ca
* **build:** add root makefile and common targets 3be185d
* **build:** put image base OS at the end of the image tag 7b8f96f
* **docs:** add readme note about monorepo paths b20f200
* **docs:** notes about bundling UI, ONNX_WEB paths for server b22f50f
* **gui:** add labels to dropdowns d3f4607
* **gui:** get default params and prompt from config 561fcb4
* **gui:** get platforms and schedulers from server ce06837
* **gui:** load models from server 46e047b
* **gui:** make an image card component showing params b5d67b4
* **gui:** move platform selector outside of mode tabs 45a097a
* **gui:** set up react-query for better request handling b13d46c
* **gui:** show recent image history 764a097
* **image:** add preliminary container files 8f77bb8
* return json struct with output path instead of image data, load images from outputs endpoint 4668841


### Bug Fixes

* **api:** add numpy version restriction to requirements ca0da31
* **api:** add recommended venv name to git ignore 5a82f39
* **api:** improve image layer order 5482978
* **api:** match model and output paths from readme c036c6f
* **api:** match model path from readme 6004f76
* **build:** add base OS to image tags 11e61d0
* **build:** add stub API unit test 1cd3bd8
* **build:** correct base OS for GUI nginx image 91c6f47
* **build:** correct output filename for JS ep 9a10f52
* **build:** install coverage in python build 32c7701
* **build:** make image suffix part of the name 150a81e
* **build:** pull GUI images from hub 03362f3
* **build:** remove bash shebang e900479
* **build:** remove unittest from CI install list, add coverage output to git ignore 16b7bde
* **build:** run GUI image build in correct subdir 86a3968
* **build:** use correct path for image script 7dcb0d1
* **build:** use CPU version of pytorch in hypothetical alpine API image af40bff
* **docs:** add numpy to install list 7f90461
* **docs:** add section about custom config a0172f8
* **docs:** describe current features 33eb7cd
* **docs:** explain running containers 08270f2
* **docs:** note DirectML for Windows 9a5ec9c
* **docs:** note numpy version requirements in readme 1f26858
* **docs:** update readme to use nginx image for GUI 37253cc
* **gui:** copy bundle to correct path within nginx image 225f5f1
* **gui:** correct paths in nginx image 7f23711
* **gui:** dedupe query lists into a component 1c9eed3
* **gui:** make more space for scheduler in image card 7c08c4b
* **gui:** prevent dropdown border from overlapping with label 26e886b
* **gui:** remove paragraph wrapping image details 0376499
* **gui:** run dev server from node image ee6cf50
* **gui:** show parameters in a grid in the image card a950343
* **gui:** switch default API host to localhost 5f1bb4a
* **gui:** switch default platform to AMD 29c4908
* **lint:** style issues in gui cd36172
