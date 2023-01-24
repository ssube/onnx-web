# Changelog

All notable changes to this project will be documented in this file. See [commit-and-tag-version](https://github.com/absolute-version/commit-and-tag-version) for commit guidelines.

## [0.5.0](https://github.com/ssube/onnx-web/compare/v0.4.0...v0.5.0) (2023-01-24)


### Features

* add additional Real ESRGAN models, strings for them ([d52c22e](https://github.com/ssube/onnx-web/commit/d52c22e58b4b62463c3d14bbbe930f1b90640209))
* add fill color control to inpaint ([3679735](https://github.com/ssube/onnx-web/commit/3679735d86e982c18ae9534118567b2a24990868))
* add outscaling option ([8d3ebed](https://github.com/ssube/onnx-web/commit/8d3ebede5a402499f285f1b06d367c8e9b8b2126))
* add ROCm provider to list ([#10](https://github.com/ssube/onnx-web/issues/10)) ([3bcd7a8](https://github.com/ssube/onnx-web/commit/3bcd7a8156328dbc0087efc12450abf22df443fd))
* add upscale controls to client, params to server ([d1e4fa9](https://github.com/ssube/onnx-web/commit/d1e4fa9cf1da5188b71d34d8a1753ba0e6186009))
* add upscaling tab and endpoint ([4aeee60](https://github.com/ssube/onnx-web/commit/4aeee60b19ac843e703f1eaf7ca50353eb386ee4))
* add version check to parameters ([be3a17b](https://github.com/ssube/onnx-web/commit/be3a17b2ffa9f78ab96a9a1659ce22a0ab0279d2))
* **api:** add basic upscaling ([77cb84c](https://github.com/ssube/onnx-web/commit/77cb84c60eb3272cb16525c4f6157367342b1726))
* **api:** add conversion script for models ([e59449f](https://github.com/ssube/onnx-web/commit/e59449fec1137cd1e8dfe688330b86f6154f9da0))
* **api:** add ESRGAN/GFPGAN deps ([9f43837](https://github.com/ssube/onnx-web/commit/9f4383716e04464825e032b0ead3c4efc1cbf64f))
* **api:** add ONNX implementation of Real ESRGAN net ([9519fc1](https://github.com/ssube/onnx-web/commit/9519fc16e993f314118cd4c9597e395fccc4300b))
* **api:** add option for HuggingFace token in convert script ([45a3ddc](https://github.com/ssube/onnx-web/commit/45a3ddc2a9dd2a5ccacf8bace8d681f16f8d81f7))
* **api:** add strength param to inpaint, remove same from upscale ([5ba752e](https://github.com/ssube/onnx-web/commit/5ba752e526ff4cb04baceeb8ee97bc9961350345))
* **api:** add support for Stable Diffusion models to conversion script ([decb281](https://github.com/ssube/onnx-web/commit/decb2813c6078664661138273d34f4e0b90ea6fb))
* **api:** backend support for multiple GPUs in diffusion pipelines ([a868c8c](https://github.com/ssube/onnx-web/commit/a868c8cf6bd249571700b0d1fbf6b2ea2cc54d09))
* **api:** return all types of models ([ee6308a](https://github.com/ssube/onnx-web/commit/ee6308a0918a865bfde2615d9a75e86f8796bb89))
* **api:** split up test scripts for diffusers and real esrgan ([48963fa](https://github.com/ssube/onnx-web/commit/48963fa591413905403adc343b8ac2124ec0eca5))
* **api:** start adding model sources to convert script ([4d0898a](https://github.com/ssube/onnx-web/commit/4d0898a52cf95566101a7085474d0607bdc6872a))
* **build:** add DirectML and ROCm images ([b18567c](https://github.com/ssube/onnx-web/commit/b18567ca43fab9408730e08926b0eeb43fb7ff8e))
* **build:** compile ONNX runtime with ROCm support ([a8bc371](https://github.com/ssube/onnx-web/commit/a8bc3714785bfc9b495406ef16366142c02f3c7a))
* **build:** run convert script when container starts ([a8769a5](https://github.com/ssube/onnx-web/commit/a8769a5919ead33d3328ff75ca09bc68168933ab))
* **docs:** add platform/model compatibility list ([b22f156](https://github.com/ssube/onnx-web/commit/b22f15600bb8f80a6c906fb25471d1f4a2ff7314))
* **gui:** add API server to settings ([d402db8](https://github.com/ssube/onnx-web/commit/d402db85092b9923b7a16b7b26ee1c035809ab4b))
* **gui:** add blend strength to inpainting controls ([691aeab](https://github.com/ssube/onnx-web/commit/691aeabfd93dc6df801306c8dee37d0a8be6e7d2))
* **gui:** add invert button to inpaint mask (fixes [#65](https://github.com/ssube/onnx-web/issues/65)) ([9e31445](https://github.com/ssube/onnx-web/commit/9e31445ccc36f9a47496331d307678faf793cb9a))
* **gui:** add menus for upscaling and correction models ([0080d86](https://github.com/ssube/onnx-web/commit/0080d86d91fbb476684a0e8614a3950c25cede06))
* **gui:** add validation to numeric inputs, token counter to prompt ([a1b16bb](https://github.com/ssube/onnx-web/commit/a1b16bb435f1890d83175ad7317bc42693efc15e))
* **run:** add Docker Compose files for API containers ([2a6df0f](https://github.com/ssube/onnx-web/commit/2a6df0f3aa52585b6dfb15c02bcb2cf66160c388))


### Bug Fixes

* **api:** actually return the filtered list of platforms ([facd69f](https://github.com/ssube/onnx-web/commit/facd69f452d5dc921da5a220c4b515833e443cb3))
* **api:** add an option to skip certain models during conversion ([556d5b8](https://github.com/ssube/onnx-web/commit/556d5b84d6b7e969b4249a6f2819959e2c571843))
* **api:** add extra models to convert script ([e083411](https://github.com/ssube/onnx-web/commit/e0834110fc31e9d8f42b5a2b1e624a7fdf425ad4))
* **api:** add missing params to load stub ([fe65746](https://github.com/ssube/onnx-web/commit/fe657468bf3ae6011690eac1de07a1842dc9e82f))
* **api:** check if output file exists for ready endpoint ([#57](https://github.com/ssube/onnx-web/issues/57)) ([b2e7ad5](https://github.com/ssube/onnx-web/commit/b2e7ad599ff0e866ea729538db8ab6a93b6976bb))
* **api:** check image size before blending ([08dbc0c](https://github.com/ssube/onnx-web/commit/08dbc0c7380c5110361d2be19c5c684b96a3266e))
* **api:** convert back to PIL after upscaling ([45d65d1](https://github.com/ssube/onnx-web/commit/45d65d1342d9d1b63f0aee06fc760c0bc9ebff40))
* **api:** convert image to numpy before upscaling ([1fe6fa9](https://github.com/ssube/onnx-web/commit/1fe6fa91fb051f9f82aa63e6b1e8d5d7c6a52587))
* **api:** copy checkpoints into correct location, handle more models ([353a655](https://github.com/ssube/onnx-web/commit/353a65513f273b5530a04fb55dc9ef85b4b57948))
* **api:** correct denoise max, add missing face strength param ([227056d](https://github.com/ssube/onnx-web/commit/227056d97686480a41b1fb5d8f282e26e0737956))
* **api:** correct name for kwargs ([9bff64c](https://github.com/ssube/onnx-web/commit/9bff64c7c9e1cad7440e6484550ff3a4006974bf))
* **api:** correct output paths, read strength from params ([a76793d](https://github.com/ssube/onnx-web/commit/a76793d1058fce99f7abfbe4414834b68bdce0f7))
* **api:** correct stub method name ([f493246](https://github.com/ssube/onnx-web/commit/f493246192ae33218ae9c0afb1516afded54cf73))
* **api:** dedupe models after removing extension ([c0ca7cf](https://github.com/ssube/onnx-web/commit/c0ca7cf62f85eb7a8c16d9d9ddcf8c2a39017d5e))
* **api:** enable tiling when fixing faces after upscaling ([ba3eff5](https://github.com/ssube/onnx-web/commit/ba3eff5c038e05be682eab126e1cbdc194ac5a14))
* **api:** explicitly delete pipeline results after saving ([dddadfc](https://github.com/ssube/onnx-web/commit/dddadfc9a256f1cd8ebbcd9313618b70a61dc7ce))
* **api:** filter platforms based on available providers (fixes [#69](https://github.com/ssube/onnx-web/issues/69)) ([c768cd8](https://github.com/ssube/onnx-web/commit/c768cd8f42d31787160cff5751b90f7cbbe2d63d))
* **api:** generate correct latents for non-square images ([86fb2ae](https://github.com/ssube/onnx-web/commit/86fb2ae28e2cdd2cd04d29a6095cdb0718d81cfb))
* **api:** get all server paths from environ ([4809e00](https://github.com/ssube/onnx-web/commit/4809e009820dc28768edc5e2242e869b9a281541))
* **api:** get ESRGAN/GFPGAN paths from server context, clean up test scripts ([120056f](https://github.com/ssube/onnx-web/commit/120056f878593989d83fdf0c5f8f59bcf8f8d0bc))
* **api:** get upscale params from request ([1f0c19a](https://github.com/ssube/onnx-web/commit/1f0c19af04e403f0ae78181544d0daddc4a967f6))
* **api:** handle parameters correctly when list of valid values is empty (fixes [#72](https://github.com/ssube/onnx-web/issues/72)) ([2921eba](https://github.com/ssube/onnx-web/commit/2921eba1f641891313daa9df7717f8c014296db5))
* **api:** include model scale ([dba6113](https://github.com/ssube/onnx-web/commit/dba6113c0983d1796b6550e7435bdb66897a9852))
* **api:** join globs to avoid py 3.10-only args ([0273dea](https://github.com/ssube/onnx-web/commit/0273dea2a68196f60a352da06aa10d7d14c3a322))
* **api:** leave pipelines on default device unless specified ([505cacf](https://github.com/ssube/onnx-web/commit/505cacfbc2a588f4890906268fed8e6062b9fa1c))
* **api:** load upscaling model from models dir ([806503c](https://github.com/ssube/onnx-web/commit/806503c7091f5c082bd65c24aa2a25bce48833c2))
* **api:** look up noise coordinates correctly ([1283bc3](https://github.com/ssube/onnx-web/commit/1283bc3d3f8ba40bd6c77ddb7bbe750b33456647))
* **api:** pass hardware platform to upscaling pipeline ([#77](https://github.com/ssube/onnx-web/issues/77)) ([f319e6a](https://github.com/ssube/onnx-web/commit/f319e6a49b7ad5f27e7410b6e77a51756782a116))
* **api:** pass image size to upscale job ([cf6a151](https://github.com/ssube/onnx-web/commit/cf6a15154890aa5180dc8826f83af04b7969893c))
* **api:** pass model to ONNX instances ([d406cd4](https://github.com/ssube/onnx-web/commit/d406cd4e99c92955c5e8fc5be2b2a3d4483d5e2d))
* **api:** pass txt2img dimensions in correct order ([be16f33](https://github.com/ssube/onnx-web/commit/be16f33151760ec64fccdf7b245cd0cd11e1000b))
* **api:** pass upscale params when creating RESRGAN ([091c4e6](https://github.com/ssube/onnx-web/commit/091c4e6109e5e276ac59228df0c091fc9e583477))
* **api:** premultiply noise before compositing ([b496e71](https://github.com/ssube/onnx-web/commit/b496e7121ce04e60b8126d5b6b59a91c51b84630))
* **api:** put conversion RNG on training device ([#67](https://github.com/ssube/onnx-web/issues/67)) ([abc1ae5](https://github.com/ssube/onnx-web/commit/abc1ae511245649e3c7498e1564f5330b72c71b0))
* **api:** report accurate image size when upscaling ([9a2e7ad](https://github.com/ssube/onnx-web/commit/9a2e7adfb89a96a91caeb2436c29873c9b70dc45))
* **api:** report accurate sizes ([4bf6875](https://github.com/ssube/onnx-web/commit/4bf68759d72b089082ddd490b110f91f0c4e5de9))
* **api:** resolve face correction model relative to model path ([5a01fe4](https://github.com/ssube/onnx-web/commit/5a01fe4cb0c40fac4bf7511a716f5be6badf8302))
* **api:** return structured error when image parameters are missing (fixes [#76](https://github.com/ssube/onnx-web/issues/76)) ([b62c7d3](https://github.com/ssube/onnx-web/commit/b62c7d3742a014f44ce199b78b6f70dc8e29844b))
* **api:** run GC after changing pipeline ([#58](https://github.com/ssube/onnx-web/issues/58)) ([4a3bb97](https://github.com/ssube/onnx-web/commit/4a3bb9734218ede79f632b208a54da510d01ea00))
* **api:** skip upscaling if scale is 1 ([b7c85aa](https://github.com/ssube/onnx-web/commit/b7c85aa51b36a3a01a369835629b745604515855))
* **api:** sort models without discarding ([b09feda](https://github.com/ssube/onnx-web/commit/b09feda47427a6a2d79c7006a5d3d6f1cab7bc8d))
* **api:** trim model names relative to model path ([4472a6f](https://github.com/ssube/onnx-web/commit/4472a6fd24fe59359009499b2f105d9ee2168c81))
* **api:** unload old model before loading next one ([9e26ee5](https://github.com/ssube/onnx-web/commit/9e26ee5b8541317c44d471a498261a2053628ddc))
* **api:** use correct base path for debug images ([634d2e8](https://github.com/ssube/onnx-web/commit/634d2e8ce63f57b108c09a8c2faa32274e4a5d1a))
* **api:** use correct coordinate system for outpainting ([a5d3ffc](https://github.com/ssube/onnx-web/commit/a5d3ffcc737e76ad8acacfded05927782737172c))
* **api:** use correct scale for background correction ([073ff8e](https://github.com/ssube/onnx-web/commit/073ff8e02f993ea3348a9564a94bda1a04ad514d))
* **api:** use training device when loading Real ESRGAN model ([#67](https://github.com/ssube/onnx-web/issues/67)) ([8c9c99e](https://github.com/ssube/onnx-web/commit/8c9c99eeb549e3748465f2cf8bbfdca1b62d7508))
* **api:** validate request params better, esp model path ([876b54a](https://github.com/ssube/onnx-web/commit/876b54a7a87d6610a1ea0223c5e315e73d6aa2f1))
* **build:** add cv2 deps to CPU image ([77d68bf](https://github.com/ssube/onnx-web/commit/77d68bf6ac7fe27b2357da90d4fd0356fa3dc8b9))
* **build:** add cv2 deps to CUDA and ROCm images ([52484e6](https://github.com/ssube/onnx-web/commit/52484e6e1f85973ec2eeb69cd64288340a814ad6))
* **build:** add launcher script and use for API images ([88f9b62](https://github.com/ssube/onnx-web/commit/88f9b624ec6bd844209e9aadc38b9dc23702d5c1))
* **build:** add params file to other images ([5286c4f](https://github.com/ssube/onnx-web/commit/5286c4f596e6ffc027107cfc707f4c9aa341f858))
* **build:** correct venv path in containers ([348a4e2](https://github.com/ssube/onnx-web/commit/348a4e2405529922b5db3b1b109f06aedb771d51))
* **build:** install prebuilt ROCm ORT ([5c026c4](https://github.com/ssube/onnx-web/commit/5c026c43d4336a527a8b318bb321532c2b4085b2))
* **build:** remove DirectML container until a package can be found ([c7bcc62](https://github.com/ssube/onnx-web/commit/c7bcc62a3ebb8ec8cbeaff82e60ac00d543a89c7))
* **build:** share layers from main image in feature branches ([455bfdd](https://github.com/ssube/onnx-web/commit/455bfddbc112fcc24b2f7dafee70912daebee511))
* **build:** use and cache venv for API job ([46e0fe2](https://github.com/ssube/onnx-web/commit/46e0fe2cf72795efd63c196fa0e1521973fe2fbe))
* **build:** use cached docker images to avoid rebuilding layers ([3f1bc0e](https://github.com/ssube/onnx-web/commit/3f1bc0e1b0fe64cc34d911a70e37fb3b4ac9e874))
* **docs:** add podman rocm command to admin guide ([fe9206c](https://github.com/ssube/onnx-web/commit/fe9206c894ec250e31d05347c1dc92bde22bc595))
* **docs:** add server admin guide, cross-link with user guide ([5d0aa60](https://github.com/ssube/onnx-web/commit/5d0aa60f1b6a3e28b681c6474230133b335df0a4))
* **docs:** list correct packages in readme, move errors to user guide ([b1e7ab0](https://github.com/ssube/onnx-web/commit/b1e7ab0a3e51baaddc276857c3a25f92d379f39c))
* **docs:** note that image parameters are not persisted when reloading ([700d31e](https://github.com/ssube/onnx-web/commit/700d31e2b87674f4063c4cc361d6e7e1bc81c6a5))
* **gui:** add server version indicator to settings ([7b49b55](https://github.com/ssube/onnx-web/commit/7b49b551d56ca51bb5a5b0658fbc545228b46a4e))
* **gui:** align background image when it is larger than canvas ([99982c6](https://github.com/ssube/onnx-web/commit/99982c6a1920f2d4683477ca37b92fb61398c311))
* **gui:** bump state version for new output path ([246aa3d](https://github.com/ssube/onnx-web/commit/246aa3dd15431ed9b6137f52b1b9403dca4613dd))
* **gui:** correct labels for Nvidia platforms ([0afd25f](https://github.com/ssube/onnx-web/commit/0afd25f25b69bebddc4ae0833cdf6d022e8f61dd))
* **gui:** correct menu state for additional models ([7cd0345](https://github.com/ssube/onnx-web/commit/7cd0345cd237f3541c40eb95f44d7d11c34d95ab))
* **gui:** correct state for face correction button ([3a3e92a](https://github.com/ssube/onnx-web/commit/3a3e92abb6e5a9609b12ff868fc9656de5dbdae1))
* **gui:** disable face correction strength slider when option is not selected ([7c60621](https://github.com/ssube/onnx-web/commit/7c60621fe2f0a2683604ec2c4f55574be1de85f1))
* **gui:** do not persist upscale image source ([dc84bec](https://github.com/ssube/onnx-web/commit/dc84becf693d6acb6b4d23c0059b02b8d7103d99))
* **gui:** draw single clicks and avoid blending mask with itself ([5e23f84](https://github.com/ssube/onnx-web/commit/5e23f84e5a20053f935e9be9eef9d87c028fc926))
* **gui:** emphasize the generate buttons, align fill color picker ([64663f5](https://github.com/ssube/onnx-web/commit/64663f5eeca8d2f37680c38b08a5db1adc405308))
* **gui:** make all image cards show at the default size ([20ed8f6](https://github.com/ssube/onnx-web/commit/20ed8f60dc0e8dde61627527025e10b12cf871f9))
* **gui:** mask canvas should not tile small images ([f1484dc](https://github.com/ssube/onnx-web/commit/f1484dc075fca73bf8a908d8c06b4ced945f0957))
* **gui:** only enable generate buttons after image sources exist ([#64](https://github.com/ssube/onnx-web/issues/64)) ([4898197](https://github.com/ssube/onnx-web/commit/4898197f7738c17c9ac19be9ccdbfa400e0f2939))
* **gui:** only send upscaling params when it is enabled ([5d2c22a](https://github.com/ssube/onnx-web/commit/5d2c22a64ae75b465bea763317bd4cb7ba5b02e6))
* **gui:** only show inpaint image once ([d6f2c62](https://github.com/ssube/onnx-web/commit/d6f2c626f67faba361fbdbb7ba64d0604f85e5dc))
* **gui:** populate empty select menus with first valid value ([0d1f236](https://github.com/ssube/onnx-web/commit/0d1f23609601f3c9aeab9268b9002a23a5bc6007))
* **gui:** prevent client image history from accumulating indefinitely ([df7bba4](https://github.com/ssube/onnx-web/commit/df7bba47dd892db9b590ebcf5e31c468f3c59e04))
* **gui:** read image size from its own field ([4d6560a](https://github.com/ssube/onnx-web/commit/4d6560aaba1a0964193ea5c8d72408dc4ed72296))
* **gui:** reduce rendering when adjusting image controls ([4615614](https://github.com/ssube/onnx-web/commit/4615614e5e22ee58006c2751a9b3725ebb6b4c70))
* **gui:** remove unused strength param from upscale ([d2c7fa9](https://github.com/ssube/onnx-web/commit/d2c7fa97e79267b4694556e0019c0abd63bd229a))
* **gui:** send blend strength for inpainting ([521fa88](https://github.com/ssube/onnx-web/commit/521fa88e0598196fda5ee28e65485e81adf08bda))
* **gui:** send upscale params ([5e5d748](https://github.com/ssube/onnx-web/commit/5e5d748c0b8f69eae27ce77e58b7f31d64918c4a))
* **gui:** set a reasonable timeout on the initial params fetch ([50fe17b](https://github.com/ssube/onnx-web/commit/50fe17bce15b597dcc53351969eb10ae9af71c72))
* **gui:** set initial fill color ([ce11165](https://github.com/ssube/onnx-web/commit/ce11165d0f72f2b63383bbaabf0e5dd9ff86e482))
* **gui:** swap toggle buttons for decent checkboxes ([46026d9](https://github.com/ssube/onnx-web/commit/46026d9aa0b22eed0addc861ff7d9c3fdadcf098))
* **gui:** use blur event on fill color for better perf ([b66cb8f](https://github.com/ssube/onnx-web/commit/b66cb8fcd6e9953ba939ab1d7b1775dd1df68151))
* move output path out of API route (for [#7](https://github.com/ssube/onnx-web/issues/7)) ([cb005d3](https://github.com/ssube/onnx-web/commit/cb005d3b5d6c5360ddbc3ad3ecd4f0cfcae486b4))
* send missing model params, add face strength control ([0e27cc8](https://github.com/ssube/onnx-web/commit/0e27cc830d8917ba083b255df667dacd17db0737))

## [0.4.0](https://github.com/ssube/onnx-web/compare/v0.3.0...v0.4.0) (2023-01-15)


### Features

* add gaussian multiply to mask filters ([40080ad](https://github.com/ssube/onnx-web/commit/40080ad46b4e0b846c92daf8e7e049bb171eaa55))
* add noise source with solid color ([5bb3f6c](https://github.com/ssube/onnx-web/commit/5bb3f6c558583fc7a135c36efb851a64646d7a57))
* **api:** add a mask filter to blend outpainting edges ([6c07d12](https://github.com/ssube/onnx-web/commit/6c07d124e0ce287b22baeaf15cba383f899d0adc))
* **api:** add endpoints for blend and noise settings ([a8f0a7a](https://github.com/ssube/onnx-web/commit/a8f0a7a7eb7e6f9ae62414c0dea034266aa8e3b3))
* **api:** add endpoints to serve GUI bundle ([88fde63](https://github.com/ssube/onnx-web/commit/88fde63e07f56a2312e38b8f87afb077368101f0))
* **api:** add helper to expand images for outpainting ([66dc532](https://github.com/ssube/onnx-web/commit/66dc5322d002c6fd2855f3594b3af39f30d86aad))
* **api:** add normal and uniform noise sources ([9376de8](https://github.com/ssube/onnx-web/commit/9376de880ec24a022e24a231e1b0314e3891f986))
* **api:** add original source and gaussian blur noise sources ([77470a6](https://github.com/ssube/onnx-web/commit/77470a610a240bdd38d5a5dc3b8842203ec48755))
* **api:** add parameters for noise source and blend op to inpaint ([e403980](https://github.com/ssube/onnx-web/commit/e403980a44cdec152f3eceae9320c3976717c5a7))
* **api:** add ready endpoint to check output status ([55e8b80](https://github.com/ssube/onnx-web/commit/55e8b800d288379fa52caa50e7d5e89cad1ab811))
* **api:** blend outpainting border with noise ([1e24018](https://github.com/ssube/onnx-web/commit/1e24018b577cc199011ea007b0371b16bd4ca349))
* **api:** limit simultaneous image workers (fixes [#15](https://github.com/ssube/onnx-web/issues/15)) ([e872eea](https://github.com/ssube/onnx-web/commit/e872eeacec63da96c30b22253d8bfc4f2be6f344))
* **api:** move txt2img into a background task ([0ef4d60](https://github.com/ssube/onnx-web/commit/0ef4d60b04fa56f7dd466f034973626a4b1df981))
* **api:** put all image pipelines on background tasks ([7e35b7b](https://github.com/ssube/onnx-web/commit/7e35b7b34f7e94488fdb14f6cb6bae3775aa9845))
* **api:** take outpainting dimensions from query params ([d9bbb9b](https://github.com/ssube/onnx-web/commit/d9bbb9bb5a92dbe933ed44feca395ebf441c331f))
* **build:** embed GUI bundle in API containers ([6eaf92a](https://github.com/ssube/onnx-web/commit/6eaf92a7486554c01837c8d364329dc10f76c3e7))
* **common:** add pod definitions for API ([e0589e2](https://github.com/ssube/onnx-web/commit/e0589e2809655d9e4ec1b2c82563b399d90dbc39))
* **gui:** add copy to source buttons to image card ([028d39c](https://github.com/ssube/onnx-web/commit/028d39c808b4c47eed060c94bc13dc4505d3dbd0))
* **gui:** add error message when server is not available (fixes [#48](https://github.com/ssube/onnx-web/issues/48)) ([65f2f4d](https://github.com/ssube/onnx-web/commit/65f2f4d953e48d5ed2294fbde06021dbc98eab1f))
* **gui:** add fill with white, toggle for outpainting ([0d53fdf](https://github.com/ssube/onnx-web/commit/0d53fdfe5308d2dbd98ee92763f18181f8e10050))
* **gui:** add menus for noise source and blend mode ([d3ad43b](https://github.com/ssube/onnx-web/commit/d3ad43bef4914df77a95e65ff353e3f7862df6e3))
* **gui:** add outpainting dimension controls to inpaint tab ([9e2921d](https://github.com/ssube/onnx-web/commit/9e2921d3de6bf782d0c25efd15122d09de199341))
* **gui:** add outpainting to API client and state ([6cd98bb](https://github.com/ssube/onnx-web/commit/6cd98bb96002f3177c08eeec4fa174b1cb7156ae))
* **gui:** add selector for mask filter ([2a30a04](https://github.com/ssube/onnx-web/commit/2a30a04e460089dadd7cb2cf167aab1a729ebf7f))
* **gui:** add slider for brush strength (fixes [#30](https://github.com/ssube/onnx-web/issues/30)) ([56ac6c6](https://github.com/ssube/onnx-web/commit/56ac6c6bc753f401077e2e6004169b482c6e39be))
* **gui:** add update instructions to error screen ([f00f36b](https://github.com/ssube/onnx-web/commit/f00f36b5b154b3683ab5ef859c142dacafa3bf8d))
* **gui:** get API root from query string if available ([f834997](https://github.com/ssube/onnx-web/commit/f83499763a510f1bc6984f9e33c997babd946a7b))
* **gui:** implement image polling on the client ([c36dadd](https://github.com/ssube/onnx-web/commit/c36daddf66f19dbeab58034b8b7c35b85eca5706))
* **gui:** produce noise based on source image histogram ([b24b1eb](https://github.com/ssube/onnx-web/commit/b24b1eb96190ee969f9aa6f9c8320a9c949b312b))
* **gui:** replace mask to grayscale with fill button ([3ad3299](https://github.com/ssube/onnx-web/commit/3ad3299734ea189f57ccebda2f35b0c8beecadb7))
* **gui:** save source and mask images while changing tabs ([4e82241](https://github.com/ssube/onnx-web/commit/4e82241491504412d45022dca3c045103104266e))
* **gui:** show mask brush preview (fixes [#39](https://github.com/ssube/onnx-web/issues/39)) ([a87dc45](https://github.com/ssube/onnx-web/commit/a87dc451fd10476a76cd23c0a1be57ef628f94f3))
* **gui:** show source behind mask with offscreen painting ([e915ab5](https://github.com/ssube/onnx-web/commit/e915ab5b8d204cdab53af9e798fcee74be790aa6))
* **gui:** split mask canvas into its own component ([1183216](https://github.com/ssube/onnx-web/commit/1183216a83f977309c75628cfc2ab9ff595e22a9))
* move API routes under prefix ([b477a99](https://github.com/ssube/onnx-web/commit/b477a9937ce11d3ce34a8165cc1f02da3bbd404a))


### Bug Fixes

* **api:** add missing origin argument to noise sources ([4675f89](https://github.com/ssube/onnx-web/commit/4675f89bb7c17153c4e6e4fc0d513ce6224ce0c5))
* **api:** add time to filenames (for [#55](https://github.com/ssube/onnx-web/issues/55)) ([16108ae](https://github.com/ssube/onnx-web/commit/16108ae1724debd8edba997999c467232a4aa741))
* **api:** always apply mask filter for inpainting ([e4020cf](https://github.com/ssube/onnx-web/commit/e4020cf3f6d0817ba04eb30f17c7cf1b587d31b7))
* **api:** blend source and noise in correct order ([eedea93](https://github.com/ssube/onnx-web/commit/eedea93adeab356422a349ff785fc1e30f157b69))
* **api:** clean up background jobs once they are ready (for [#55](https://github.com/ssube/onnx-web/issues/55)) ([9c93e16](https://github.com/ssube/onnx-web/commit/9c93e16698d2277026b76dd56068bcfd55de53cb))
* **api:** convert mask before blending source ([f142418](https://github.com/ssube/onnx-web/commit/f1424187077eceafbe5a49f1f80dfdb75726aa05))
* **api:** correct endpoint name for blend ops ([0ed11af](https://github.com/ssube/onnx-web/commit/0ed11af34bc81d40734b71393e93e8740e4ad4b1))
* **api:** correct type of provider in output path ([b1aca92](https://github.com/ssube/onnx-web/commit/b1aca928ab2b384689f5138235db34644083eb78))
* **api:** fill gaussian blur with noise first ([e2d17e1](https://github.com/ssube/onnx-web/commit/e2d17e18335cdb6d53b4abfa13e96ed94dd63bf0))
* **api:** get default params from file, enforce minimum params ([e8b580a](https://github.com/ssube/onnx-web/commit/e8b580a5deaa661009e2b7084a5db8fb30093611))
* **api:** include all parameters in output path ([e429baf](https://github.com/ssube/onnx-web/commit/e429bafeef04cb581a91ad2b3d21aa018f7cde63))
* **api:** limit outpainting using image size params ([34fa3f6](https://github.com/ssube/onnx-web/commit/34fa3f6341c1e239c6e57aa59238e284b2ace9ce))
* **api:** make all path vars relative to API dir ([360a151](https://github.com/ssube/onnx-web/commit/360a1518676af9ab7bc52de47bb5f0826b4dce23))
* **api:** reduce copies, fix function signatures ([f5ed77a](https://github.com/ssube/onnx-web/commit/f5ed77a349e1b6a785334a5fbb4b521174eec794))
* **api:** resize images after getting request params ([c29c92e](https://github.com/ssube/onnx-web/commit/c29c92ed90a46b1c7b335ca5b4b6b4e3adfd4585))
* **api:** restore inpainting without outpainting ([09c9b2c](https://github.com/ssube/onnx-web/commit/09c9b2c028363d7a03045abd905d67525fc9b41a))
* **api:** reuse results of blur modes ([ef06b45](https://github.com/ssube/onnx-web/commit/ef06b4559936d1ba8ef67c6ed64af966474883f5))
* **api:** send CORS more consistently ([fa82ac1](https://github.com/ssube/onnx-web/commit/fa82ac18ab9eb4dd3549b28061fba8642f4e69e8))
* **api:** set default mask filter to none, matching the client ([df6b071](https://github.com/ssube/onnx-web/commit/df6b07194b3a8dd2fa4bbc040946de26a2ab56b6))
* **api:** use correct param name for platform ([a3029c3](https://github.com/ssube/onnx-web/commit/a3029c30a603c36a3aeb09543e01c146e0720cce))
* **api:** use full-image compositing, write debug images to output dir ([081a96d](https://github.com/ssube/onnx-web/commit/081a96d536b50255f91f1597a66f1de15de3d126))
* **api:** use same parameter name as GUI for negative prompt ([dc33b7c](https://github.com/ssube/onnx-web/commit/dc33b7c8876a33bd2b5891268c24d5b262cac68d))
* **build:** correct path to pip requirements ([f46647c](https://github.com/ssube/onnx-web/commit/f46647cf6dd09550b87a4bb0be8eab7002e33648))
* **build:** correct paths for GUI bundle ([6c11f52](https://github.com/ssube/onnx-web/commit/6c11f52006b5251b9743ece6e70e285cf3ae3af0))
* **build:** install torch before other packages in CPU container ([e025dbb](https://github.com/ssube/onnx-web/commit/e025dbb87d40802736ae9d60d0601898b3eab6b7))
* **build:** install torch before other packages in CUDA container ([b60ccd5](https://github.com/ssube/onnx-web/commit/b60ccd506b93959b44b4e9b571c76034c93b6c5f))
* **docs:** add links to GUI client at GH pages ([47f9eb1](https://github.com/ssube/onnx-web/commit/47f9eb1391e43b0a5e59329853eda116745b0bc5))
* **docs:** note NodeJS dependency for building GUI, note about ONNX DML upgrading numpy ([74eaac3](https://github.com/ssube/onnx-web/commit/74eaac371dab335f503ee8757c98b3563fa799a3))
* **gui:** avoid saving mask while actively painting ([d5f8838](https://github.com/ssube/onnx-web/commit/d5f8838ceb5bc9bc344d326c4fc8714281014ceb))
* **gui:** break up state into slices for each tab ([689a6a1](https://github.com/ssube/onnx-web/commit/689a6a183fce968793626d262022bde629e19c04))
* **gui:** clear loading data after card leaves ([600ebae](https://github.com/ssube/onnx-web/commit/600ebae73aac30c7923f122b0652321765ac4753))
* **gui:** correct label for blur mode ([1c2c8b2](https://github.com/ssube/onnx-web/commit/1c2c8b2689c2155e27a5f5dcd190dea2ec1377bf))
* **gui:** disable react profiling in default bundle ([1bb0a3a](https://github.com/ssube/onnx-web/commit/1bb0a3aed85ccda33baad6ba62dd5e2f47369f0c))
* **gui:** do not persist loading flag ([dcfce81](https://github.com/ssube/onnx-web/commit/dcfce81fedffe7e67805f1822f0aec39687fedd4))
* **gui:** history is not iterable error when loading existing state ([7885bbf](https://github.com/ssube/onnx-web/commit/7885bbfbdd55b6d84e6dfcac50e3c365300e76f8))
* **gui:** improve API link example ([c09eb75](https://github.com/ssube/onnx-web/commit/c09eb75ab44fc18eca150e527a67bd254177b1d4))
* **gui:** improve performance while using image controls ([35e2e1d](https://github.com/ssube/onnx-web/commit/35e2e1dda643f5d44dd6a218535a4b85bc28c2e5))
* **gui:** invalidate loading query after mutations ([fa639ef](https://github.com/ssube/onnx-web/commit/fa639efff3fb9597d8214aa310a9760d0bd5a220))
* **gui:** load config relative to current page (fixes [#43](https://github.com/ssube/onnx-web/issues/43)) ([2e5c786](https://github.com/ssube/onnx-web/commit/2e5c7867a4cd644e40b1264ecf33b61182ca5b88))
* **gui:** prevent mask canvas from going into a save loop ([4dc915d](https://github.com/ssube/onnx-web/commit/4dc915d5c93dcce7ab6b3cc7035bd3c6fbb1500e))
* **gui:** restore delete image button ([68eb8eb](https://github.com/ssube/onnx-web/commit/68eb8eb9b2dd87fe17c9f462decbe0d3cf74cd60))
* **gui:** send CFG to API as decimal ([ef33301](https://github.com/ssube/onnx-web/commit/ef33301d6b1740736e991127b5030d17bbb6dd45))
* **gui:** send strength for img2img requests ([26a8ce7](https://github.com/ssube/onnx-web/commit/26a8ce7095be43fce9da7a2cb9c396b8b4186e27))

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
