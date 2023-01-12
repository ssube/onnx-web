# Changelog

All notable changes to this project will be documented in this file. See [commit-and-tag-version](https://github.com/absolute-version/commit-and-tag-version) for commit guidelines.

## [0.3.0](https://github.com/ssube/onnx-web/compare/v0.2.1...v0.3.0) (2023-01-12)


### Features

* **api:** add inpaint endpoint ([182ce6d](https://github.com/ssube/onnx-web/commit/182ce6de90361c2eb5d47861ab13570c8186db18))
* **api:** add params endpoint, defaults file ([03fd728](https://github.com/ssube/onnx-web/commit/03fd728ab049187274c7208f530eaa9755d6ae98))
* **api:** set up venv in CUDA container, add onnxruntime-gpu ([a3fe2ca](https://github.com/ssube/onnx-web/commit/a3fe2ca559e8c668b44220b41c4cdd2ada317e01))
* **build:** add github status jobs (fixes [#28](https://github.com/ssube/onnx-web/issues/28)) ([c8b2abc](https://github.com/ssube/onnx-web/commit/c8b2abc110059d71b400aec30e3746b11c1b342d))
* **build:** replace Buster image with CUDA-based Ubuntu ([07c18c2](https://github.com/ssube/onnx-web/commit/07c18c2245ec6f0063ed2278976c6704dd7f1404))
* **build:** upload pip package (fixes [#29](https://github.com/ssube/onnx-web/issues/29)) ([8452b73](https://github.com/ssube/onnx-web/commit/8452b7384cb4ee4a50b7887cf0fadbd6e2c9d684))
* **gui:** add inpaint call to API client ([15ab44f](https://github.com/ssube/onnx-web/commit/15ab44f2ad03fa691b8b9df6edc426d8f0cb71ce))
* **gui:** add inpaint tab and basic image mask component ([11b9295](https://github.com/ssube/onnx-web/commit/11b9295efc8ebe8edac445e25a92cba8e143baae))
* **gui:** display source images after selection ([f49fc96](https://github.com/ssube/onnx-web/commit/f49fc960c91ac786aacdb8662856c645da07d9bc))
* **gui:** implement mask painting, flood fill ([5e71292](https://github.com/ssube/onnx-web/commit/5e712923db6a898275b22d35d0015eec6ee567aa))
* **gui:** load and merge server params with config ([37efd51](https://github.com/ssube/onnx-web/commit/37efd513416b605c0d231432a88b292fc5275670))
* **gui:** persist image control state (fixes [#11](https://github.com/ssube/onnx-web/issues/11)) ([07fa81a](https://github.com/ssube/onnx-web/commit/07fa81a66bce91850093c9718ff4d4aed05aa2e6))
* **gui:** share image history between tabs, add setting to adjust length of history (fixes [#22](https://github.com/ssube/onnx-web/issues/22)) ([662bf42](https://github.com/ssube/onnx-web/commit/662bf42454c31df6f439d440f0f2cfe4d59397da))


### Bug Fixes

* **api:** add latents to inpaint, remove strength ([131cff6](https://github.com/ssube/onnx-web/commit/131cff6ba46dd35477333ee7998a4b304f89b4eb))
* **api:** allow decimal CFG ([2f3b5c0](https://github.com/ssube/onnx-web/commit/2f3b5c06c7cd6b0864bac94666f5aa9f02abe25f))
* **api:** omit negative prompt from pipeline ([9bb01cc](https://github.com/ssube/onnx-web/commit/9bb01cc01d47f39f3dda94bf228497f9cb70ded1))
* **api:** pass seed when calculating inpaint filenames ([d20fb91](https://github.com/ssube/onnx-web/commit/d20fb910739b24cb295e4d5c38a470f866774d5d))
* **api:** rename to avoid shadowing type ([48f42e5](https://github.com/ssube/onnx-web/commit/48f42e56fe1bcf495f7dfe4062330d83e486f664))
* **api:** use correct dict for type hints ([4abf760](https://github.com/ssube/onnx-web/commit/4abf760716fc3baef44da8da323d28939e386e97))
* **build:** add venv to CPU image ([362b732](https://github.com/ssube/onnx-web/commit/362b7327072aab7cc5ab3fcf50a3a46733db29cb))
* **build:** define template for curl jobs ([9f7e16b](https://github.com/ssube/onnx-web/commit/9f7e16b62f8b646d508314ab666e038dd5bc0dff))
* **build:** put Github status jobs in correct stages ([e704db5](https://github.com/ssube/onnx-web/commit/e704db50b1783cb6b24b32b58e2e49bc66c2a320))
* bump package versions to 0.2.1 ([760b162](https://github.com/ssube/onnx-web/commit/760b162a5578ff5c5a091e6281f55452b520d157))
* **docs:** describe how to install inpainting model ([2332c44](https://github.com/ssube/onnx-web/commit/2332c44cee05e1b98821041c6ca343cef606bea2))
* **gui:** allow decimal steps for CFG ([2ff4aee](https://github.com/ssube/onnx-web/commit/2ff4aee8873194b932e47d7c20dd8ea91396094c))
* **gui:** consistently load image controls from server params ([4a6458d](https://github.com/ssube/onnx-web/commit/4a6458d8f68e0ee262db1bbb94944d099152720a))
* **gui:** default mask brush to white/full replacement ([63758b0](https://github.com/ssube/onnx-web/commit/63758b0e21314de96f1f3f6a82ac475ea44cfcf6))
* **gui:** handle cancel from file input ([6b3c0fe](https://github.com/ssube/onnx-web/commit/6b3c0fea45e1ab8fc0ff249e125fa0c7d344e47c))
* **gui:** move seed control onto same line with cfg and steps ([1aa2181](https://github.com/ssube/onnx-web/commit/1aa2181d901b7ab86e3268fe20aac2aeeda50946))

### [0.2.1](https://github.com/ssube/onnx-web/compare/v0.2.0...v0.2.1) (2023-01-08)


### Bug Fixes

* **gui:** allow max safe seed constant ([477d89b](https://github.com/ssube/onnx-web/commit/477d89b6fc8ae61031f25573c9edaad84fcbf4d2))

## [0.2.0](https://github.com/ssube/onnx-web/compare/v0.1.0...v0.2.0) (2023-01-08)


### Features

* **api:** add image with pytorch CUDA ([a721008](https://github.com/ssube/onnx-web/commit/a721008d94d34f655d8749f9fc8fa9f801950427))
* **api:** add img2img endpoint ([09ce654](https://github.com/ssube/onnx-web/commit/09ce6546beac735369a4535646bb1362ff0eaac3))
* **api:** experimentally add CUDA provider and Nvidia platform ([8359bd5](https://github.com/ssube/onnx-web/commit/8359bd5b9969051dde4686223f602cdb77b15901))
* **api:** switch to package structure ([599e0ee](https://github.com/ssube/onnx-web/commit/599e0ee2d0df405062f660af5adae4aaa2fcdb3d))
* **api:** use hash of params as output filename ([e82379c](https://github.com/ssube/onnx-web/commit/e82379c61f7bfafaf102379cddd6fb7407dc99e2))
* **gui:** add download and delete buttons to image history ([e605c9f](https://github.com/ssube/onnx-web/commit/e605c9f66b4bbcf76b8b7ff113792c62087c2716))
* **gui:** add seed to image controls with random button (fixes [#16](https://github.com/ssube/onnx-web/issues/16)) ([4585792](https://github.com/ssube/onnx-web/commit/45857924928817fd0913af215cfed6be25903dcf))
* **gui:** add sliders to numeric inputs ([c5e0439](https://github.com/ssube/onnx-web/commit/c5e0439aa5994c2ca25828589178e3ad91c7d1cc))
* **gui:** add strength to img2img controls ([2328c5f](https://github.com/ssube/onnx-web/commit/2328c5f46a76d7d08067f9fc3a9611000cdda83f))
* **gui:** implement img2img tab ([98a8db1](https://github.com/ssube/onnx-web/commit/98a8db16a33f1e389ecf3cd9237cbdedd38d3b59))
* **gui:** put in a proper loading card ([3ec8f7c](https://github.com/ssube/onnx-web/commit/3ec8f7c2fcfc3b21d75f17b9ac31d424ce4fdd1a))
* implement negative prompts ([f2e2b20](https://github.com/ssube/onnx-web/commit/f2e2b20f18bc8ec1e452ae5ba7929f7ad7ad81f2))
* make additional schedulers available in UI ([93e53f6](https://github.com/ssube/onnx-web/commit/93e53f6dc3d312f348df156780d5f89712f1c087))


### Bug Fixes

* **api:** adjust output path for module structure ([c6662d1](https://github.com/ssube/onnx-web/commit/c6662d155da9176e799fd8979ed7be289982b593))
* **api:** clamp im2img strength ([282a7cf](https://github.com/ssube/onnx-web/commit/282a7cfa2af5ff0c02dd3a33f43858ca9f5c0b4d))
* **api:** defer first model load until first request ([0232c71](https://github.com/ssube/onnx-web/commit/0232c7180c16cc6a497b28be1d1381bd89e099c5))
* **api:** hash full range of seed values ([057eea2](https://github.com/ssube/onnx-web/commit/057eea25ef52de4f945f40a805ec2bb6cd46f9e6))
* **api:** keep strength as a float ([926f77b](https://github.com/ssube/onnx-web/commit/926f77b3b0758a16196392a505d5ca695e54422b))
* **api:** remove prompt from output name ([0d4c0a5](https://github.com/ssube/onnx-web/commit/0d4c0a5942a150cbb407a621f4a6c85eef1f41b6))
* **api:** seed rng for each image ([8c133e9](https://github.com/ssube/onnx-web/commit/8c133e9230eb0ecaefcabd3209d926603eea6903))
* **api:** typesafe param hashing ([f4ca6a0](https://github.com/ssube/onnx-web/commit/f4ca6a0547d5414da25c395b38ae628c97198964))
* **api:** update serve app name, add module entrypoint ([b59519c](https://github.com/ssube/onnx-web/commit/b59519cb7e1300e39c1e6bb7c704b701e7ffc36a))
* **build:** automatically push after tagging releases ([55d4354](https://github.com/ssube/onnx-web/commit/55d435489a2ef8d93c5082229c6bf6fd09d57201))
* **docs:** add section with known errors ([067a9b3](https://github.com/ssube/onnx-web/commit/067a9b39812f6ef9dae7255bcb8e28797aed0872))
* **docs:** begin clarifying packages for each environment ([#19](https://github.com/ssube/onnx-web/issues/19)) ([f99438e](https://github.com/ssube/onnx-web/commit/f99438e623891af122495af6cceefeb6991454a4))
* **docs:** include onnxruntime in list of common deps ([6442e68](https://github.com/ssube/onnx-web/commit/6442e68e676271a1d5ae1ddefac9e8995f48cb9d))
* **docs:** move ONNX DML package to Windows setup ([16c8b54](https://github.com/ssube/onnx-web/commit/16c8b54b1a558afe7299b7e970ddd599976b166c))
* **docs:** note cloning and fix test script name in readme ([9973bf1](https://github.com/ssube/onnx-web/commit/9973bf1bfc423194c353fbb2d50336d88dabc8de))
* **docs:** note python3 command and venv "name cmd" error in readme ([f119cb4](https://github.com/ssube/onnx-web/commit/f119cb41000784abb4ea62ea31487e4da37ed2b2))
* **gui:** add npm ignore ([8f7c1e7](https://github.com/ssube/onnx-web/commit/8f7c1e705b0329b3d537827564008033607c2e82))
* **gui:** add strings for more stable diffusion models, nvidia GPUs ([33fd5f1](https://github.com/ssube/onnx-web/commit/33fd5f1b532e76d8b0777a60e2815c5b49e1fcc5))
* **gui:** bind dev server to localhost by default, open binding in containers ([fc988e4](https://github.com/ssube/onnx-web/commit/fc988e4b5bc4d29a3a71121538f8d362490c3212))
* **gui:** disable img2img tab for now, consistent quotes in jsx ([de48450](https://github.com/ssube/onnx-web/commit/de48450730bf59177372f81e322a77754e1c8636))
* **gui:** handle decimal inputs correctly ([d5c4040](https://github.com/ssube/onnx-web/commit/d5c4040b073bbd94076ccce9f7212efe25958254))
* **gui:** key image history by order ([17e62fb](https://github.com/ssube/onnx-web/commit/17e62fb8e393bdccfc33577ba8584ac72e8bb5b1))
* **gui:** limit seed to safe values, prep for more settings ([3dfbb00](https://github.com/ssube/onnx-web/commit/3dfbb0061b02e8950d40abc2840622ac471de119))
* **gui:** send seed with img2img requests ([4894e0d](https://github.com/ssube/onnx-web/commit/4894e0ddd691cd642b7e2af91e439c603a5cc796))
* **gui:** switch txt2img to post on client ([e454203](https://github.com/ssube/onnx-web/commit/e4542031c4262b1a12be639e5bad747763eb3e8b))

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
