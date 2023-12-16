# Getting Started

onnx-web is a tool for generating images with Stable Diffusion pipelines, including SDXL.

> Welcome to onnx-web, a Stable Diffusion tool designed for straightforward and versatile use in AI art. Whether you're
> running on an AMD or Nvidia GPU, on a Windows desktop, or a Linux server, onnx-web is compatible across various
> setups. It goes a step further by supporting multiple GPUs simultaneously, and with SDXL and LCM available for all
> platforms, users can harness its capabilities without constraints. The panorama mode stands out, enabling regional
> prompts without the need for additional plugins. Get ready to explore the technical aspects of onnx-web and discover
> how it seamlessly fits into your AI art toolkit.

OR

> Welcome to onnx-web, your gateway to exploring the cutting-edge realm of Stable Diffusion in AI art. This guide is tailored for new users seeking a seamless entry into the diverse and creative possibilities offered by onnx-web. Whether you're a seasoned AI art enthusiast or just dipping your toes into the world of generative art, onnx-web provides a user-friendly yet powerful environment to unleash your creativity.

> Installation and System Requirements:
> Before diving into the creative process, it's crucial to get onnx-web up and running on your system. We offer multiple installation methods to cater to users of varying technical expertise. For beginners on Windows, an all-in-one EXE bundle simplifies the installation process. Intermediate users can opt for a cross-platform installation using a Python virtual environment, providing greater customization. Server administrators can explore OCI containers for streamlined deployment. Keep in mind the minimum and recommended system requirements to ensure optimal performance, with options for optimizations tailored to low-memory users.

> Understanding onnx-web's Core Features:
> onnx-web introduces a set of core features that form the backbone of your AI art journey. The Stable Diffusion process, capable of running on both AMD and Nvidia GPUs, powers the image generation pipeline. Explore the diverse tabs in the web UI, each offering unique functionalities such as text-based image generation, upscaling, blending, and model management. Dive into the technical details of prompt syntax, model conversions, and the intricacies of parameters, gaining a deeper understanding of how to fine-tune the AI art creation process.

> Unlocking Specialized Pipelines:
> Delve into the specialized pipelines within onnx-web, such as the panorama and highres features, each designed to elevate your creative output. The panorama pipeline allows for the generation of large and seamless images, enhanced by the utilization of region prompts and seeds. Highres, on the other hand, provides a super-resolution upscaling technique, refining images with iterative img2img processes. Learn how to harness these features effectively, understanding the nuances of region-based modifications, tokens, and optimal tile configurations.

> Optimizing Performance and Memory Usage:
> Discover how onnx-web caters to users with varying hardware configurations. Uncover optimization techniques such as converting models to fp16 mode, offloading computations to the CPU, and leveraging specialized features like the panorama pipeline for efficient memory usage. Gain insights into the considerations and trade-offs involved in choosing parameters, tile configurations, and employing unique prompts for distinct creative outcomes.

> This guide sets the stage for your onnx-web journey, offering a balance of technical depth and user-friendly insights to empower you in your AI art exploration. Let's embark on this creative venture together, where innovation meets technical precision.

## Contents

- [Getting Started](#getting-started)
  - [Contents](#contents)
  - [Setup](#setup)
    - [Windows bundle setup](#windows-bundle-setup)
    - [Cross platform setup](#cross-platform-setup)
    - [Server setup with containers](#server-setup-with-containers)
  - [Running](#running)
    - [Running the server](#running-the-server)
    - [Running the web UI](#running-the-web-ui)
  - [Tabs](#tabs)
    - [Txt2img Tab](#txt2img-tab)
    - [Img2img Tab](#img2img-tab)
    - [Inpaint Tab](#inpaint-tab)
    - [Upscale Tab](#upscale-tab)
    - [Blend Tab](#blend-tab)
    - [Models Tab](#models-tab)
    - [Settings Tab](#settings-tab)
  - [Image parameters](#image-parameters)
    - [Common image parameters](#common-image-parameters)
    - [Unique image parameters](#unique-image-parameters)
  - [Prompt syntax](#prompt-syntax)
    - [LoRAs and embeddings](#loras-and-embeddings)
    - [CLIP skip](#clip-skip)
  - [Highres](#highres)
    - [Highres prompt](#highres-prompt)
    - [Highres iterations](#highres-iterations)
  - [Profiles](#profiles)
    - [Loading from files](#loading-from-files)
    - [Saving profiles in the web UI](#saving-profiles-in-the-web-ui)
    - [Sharing parameters profiles](#sharing-parameters-profiles)
  - [Panorama pipeline](#panorama-pipeline)
    - [Region prompts](#region-prompts)
    - [Region seeds](#region-seeds)
  - [Grid mode](#grid-mode)
    - [Grid tokens](#grid-tokens)
  - [Memory optimizations](#memory-optimizations)
    - [Converting to fp16](#converting-to-fp16)
    - [Moving models to the CPU](#moving-models-to-the-cpu)

## Setup

### Windows bundle setup

1. Download
2. Extract
3. Security flags
4. Run

> onnx-web offers multiple installation methods catering to users with varying levels of expertise. For beginners on
> Windows, an accessible option is the all-in-one EXE bundle. This bundle, distributed as a ZIP archive, requires
> extraction before execution. It's crucial to disable the Windows mark-of-the-web check to ensure seamless
> functionality. Upon extraction, initiate onnx-web by running the EXE file. Once model conversion is complete, the
> server will commence, opening a browser window for immediate access to the web UI.

### Cross platform setup

Link to the other methods.

> Intermediate users can opt for a cross-platform installation using a Python virtual environment. Ensure a functioning
> Python setup with either pip or conda. Begin by cloning the onnx-web git repository, followed by installing the base
> requirements and those specific to your GPU platform. Execute the launch script, patiently waiting for the model
> conversion process to conclude. Post-conversion, open your browser and load the web UI for interaction.

### Server setup with containers

> For server administrators, onnx-web provides compatibility with OCI containers, offering a streamlined deployment
> method. Choose the installation method that aligns with your proficiency and system requirements, whether it's the
> beginner-friendly EXE bundle, the intermediate cross-platform setup, or the containerized deployment for server
> admins. Each pathway ensures a straightforward onnx-web installation tailored to your technical needs.

## Running

### Running the server

Run server or bundle.

> Initiate onnx-web by launching the Python server application, a process that demands your attention before proceeding
> further. Specifically, ensure that your models are converted before the server commences its operation. For Windows
> users, the distribution is available in the form of a convenient EXE bundle. Allow the server the necessary time to
> perform this crucial model conversion, a prerequisite for optimal functionality during your creative sessions.
> Exercise patience during this stage to guarantee a smooth experience with onnx-web.

### Running the web UI

Open web UI.

Use it from your phone.

> After the server has successfully completed its startup procedures, access the web UI through your chosen evergreen
> browser, including mobile browsers. Use the same URL and local port as the server to establish the essential
> connection. This step solidifies the link between the server and the user interface, unlocking the full capabilities
> of onnx-web. For those seeking additional functionality, explore ControlNet using your phone camera directly through
> the web UI. With these meticulous steps completed, you are now equipped to efficiently harness the power of onnx-web
> and navigate the realm of Stable Diffusion without compromise.

## Tabs

### Txt2img Tab

Words go in, pictures come out.

> The txt2img tab in onnx-web serves the purpose of generating images from text prompts. Users can input textual
> descriptions and witness the algorithm's creative interpretation, providing a seamless avenue for text-based image
> generation.

### Img2img Tab

Pictures go in, better pictures come out.

ControlNet lives here.

> For image-based prompts, the img2img tab is the go-to interface within onnx-web. Beyond its fundamental image
> generation capabilities, this tab introduces the ControlNet mode, empowering users with advanced control over the
> generated images through an innovative feature set.

### Inpaint Tab

Pictures go in, parts of the same picture come out.

> The inpaint tab specializes in image generation with a unique combination of image prompts and masks. This
> functionality allows users to guide the algorithm using both the source image and a mask, enhancing the precision and
> customization of the generated content.

### Upscale Tab

Just highres and super resolution.

> Addressing the need for higher resolutions, the upscale tab provides users with tools for high resolution and super
> resolution. This feature is particularly useful for enhancing the quality and clarity of generated images, meeting
> diverse artistic and practical requirements.

### Blend Tab

Use the mask tool to combine two images.

> Enabling users to combine outputs or integrate external images, the blend tab in onnx-web offers a versatile blending
> tool. This functionality adds a layer of creativity by allowing users to merge multiple outputs or incorporate
> external images seamlessly.

### Models Tab

Add and manage models.

> Central to managing the core of onnx-web, the models tab provides users with the capability to manage Stable Diffusion
> models. Additionally, it allows for the management of LoRAs (Latents of Random Ancestors) associated with these
> models, facilitating a comprehensive approach to model customization.

### Settings Tab

Manage web UI settings.

Reset buttons.

> Tailoring the user experience, the settings tab is the control center for managing onnx-web's web UI settings. Users
> can configure server APIs, toggle dark mode for a personalized visual experience, and reset other tabs as needed,
> ensuring a user-friendly and customizable environment.

## Image parameters

### Common image parameters

- Scheduler
  - > Role: The scheduler dictates the annealing schedule during the diffusion process.
  - > Explanation: It determines how the noise level changes over time, influencing the diffusion process to achieve the desired balance between exploration and exploitation during image generation.
- Eta
  - only for DDIM
- CFG
  - > Role: CFG is integral for conditional image generation, allowing users to influence the generation based on specific conditions.
  - > Explanation: By adjusting the CFG, users can guide the diffusion process to respond to certain prompts, achieving conditional outputs aligned with the specified criteria.
- Steps
  - > Role: Steps determine the number of diffusion steps applied to the image.
  - > Explanation: More steps generally result in a more refined image but require additional computation. Users can fine-tune this parameter based on the desired trade-off between quality and computational resources.
- Seed
  - > Role: The seed initializes the randomization process, ensuring reproducibility.
  - > Explanation: Setting a seed allows users to reproduce the same image by maintaining a consistent starting point for the random processes involved in the diffusion, facilitating result replication.
- Batch size
  - > Role: Batch size influences the number of samples processed simultaneously.
  - > Explanation: A larger batch size can expedite computation but may require more memory. It impacts the efficiency of the Stable Diffusion process, with users adjusting it based on available resources and computational preferences.
- Prompt
  - > Role: The prompt provides the textual or visual input guiding the image generation.
  - > Explanation: It serves as the creative input for the algorithm, shaping the direction of the generated content. Users articulate their artistic vision or preferences through carefully crafted prompts.
- Negative prompt
  - > Role: Negative prompts offer a counterbalance to positive prompts, influencing the generation towards desired qualities or away from specific characteristics.
  - > Explanation: By including a negative prompt, users can fine-tune the generated output, steering it away from undesired elements or towards a more nuanced and controlled result.
- Width, height
  - > Role: These parameters define the dimensions of the generated image.
  - > Explanation: Users specify the width and height to control the resolution and aspect ratio of the output. This allows for customization based on the intended use or artistic preferences.

### Unique image parameters

- UNet tile size
  - > One such parameter is the UNet tile size. This parameter governs the maximum size for each instance the UNet model runs. While it aids in reducing memory usage during panoramas and high-resolution processes, caution is needed. Reducing this below the image size in txt2img mode can result in repeated "totem pole" bodies, highlighting the importance of aligning the tile size appropriately with the intended use case.
- UNet overlap
  - > The UNet overlap parameter plays a pivotal role in determining how much UNet tiles overlap. For most high-resolution applications, a value of 0.25 is recommended. However, for larger panoramas, opting for values between 0.5 to 0.75 seamlessly blends tiles, significantly enhancing panorama quality. Balancing this parameter ensures optimal performance in diverse scenarios.
- Tiled VAE
  - > For users engaging with the tiled VAE parameter, the choice revolves around whether the VAE (Variational Autoencoder) operates on the entire image or in smaller tiles. Opting for the tiled VAE not only accommodates larger images but also reduces VRAM usage. Notably, it doesn't exert a substantial impact on image quality, making it a pragmatic choice for scenarios where resource efficiency is a priority.
- VAE tile size
  - > Parallel to UNet, the VAE (Variational Autoencoder) introduces two additional parameters: VAE tile size and VAE overlap. These mirror the UNet tile size and UNet overlap parameters, applying specifically to the VAE when the tiled VAE is active. Careful consideration of these parameters ensures effective utilization of onnx-web's capabilities while adapting to the unique requirements of your image generation tasks.
- VAE overlap

See the complete user guide for details about the highres, upscale, and correction parameters.

## Prompt syntax

> When crafting prompts in onnx-web, it's essential to note the distinctive syntax, akin to other familiar tools like
> auto1111 but with notable differences in token usage. Prompt tokens in onnx-web are encapsulated within angle
> brackets, starting with < and concluding with >.

### LoRAs and embeddings

`<lora:filename:1.0>` and `<inversion:filename:1.0>`.

> A unique feature in onnx-web is the inclusion of LoRA networks and embeddings using the lora token. The syntax
> involves specifying the network name, followed by a colon, and then indicating the desired weight. For instance,
> <lora:name:weight>. Most LoRA networks exhibit optimal performance within a strength range of 0.8 to 1.0. However,
> some networks can be effectively utilized at higher or lower values, spanning from -1.0 to approximately 5.0 for
> certain sliders. This flexibility provides users with nuanced control over the influence of LoRA networks on the
> generated images.

### CLIP skip

`<clip:skip:2>` for anime.

> Additionally, onnx-web introduces the CLIP skip functionality, which proves useful for skipping later stages of the
> CLIP keyword hierarchy. Incorporating CLIP skip with a specified level can enhance image results, particularly
> beneficial for genres like anime. Skipping 2 levels, for example, can refine the output by bypassing stages in the
> keyword hierarchy, optimizing the creative outcome. This distinct combination of tokens and functionalities allows
> users to tailor their prompts in onnx-web for precise and expressive image generation.

## Highres

Called highres because it's not the hires fix, it's more like SDXL.

> onnx-web introduces the Highres feature, a powerful tool for super-resolution upscaling followed by img2img refinement
> to restore intricate details. This feature can be iteratively applied to exponentially increase image resolution. For
> instance, using a 2x upscaling model, two iterations of Highres can produce an image four times the original size,
> while three iterations result in an image eight times the size, and so forth.
>
> The technique employed in Highres draws parallels to the hires fix seen in other tools but diverges by adopting a
> methodology akin to what SDXL utilizes. In this approach, the diffusion model refines its own output by running
> img2img on smaller tiles. This technique ensures a comprehensive refinement process, though it introduces a unique
> consideration – the img2img tiles do not perceive the entire image context. Consequently, using a distinct and more
> generic prompt during Highres iterations can be advantageous. Such prompts add more detail without introducing
> specific characters, contributing to a refined output without sacrificing the broader context.

### Highres prompt

`txt2img prompt || img2img prompt`

> One distinctive aspect of Highres is its ability to operate with its own prompt, which is separate from the base
> txt2img prompt. The demarcation is achieved using the double pipe syntax: ||. This separation allows users to finely
> control the details and characteristics emphasized during the high-resolution upscaling and refinement process.

### Highres iterations

Highres will apply the upscaler and highres prompt (img2img pipeline) for each iteration.

The final size will be `scale ** iterations`.

## Profiles

Saved sets of parameters for later use.

> The onnx-web web UI simplifies user experience with the introduction of a feature known as profiles. This
> functionality empowers users to save preferred combinations of image parameters for future use, providing a convenient
> way to replicate successful settings and streamline the image generation process. When you discover a configuration
> that consistently produces impressive images, you have the option to save it as a named profile.

### Loading from files

- load from images
- load from JSON

### Saving profiles in the web UI

Use the save button.

> The process of saving profiles involves capturing the entire set of image parameters you've fine-tuned to achieve the
> desired results. Once saved, these profiles are conveniently accessible for later usage. An additional benefit is the
> ability to download your profiles as a JSON snippet. This JSON representation encapsulates the intricacies of your
> selected parameters, allowing you to easily share your preferred configurations with others on platforms like Discord
> or various online forums.

### Sharing parameters profiles

Use the download button.

Share profiles in the Discord channel.

> onnx-web's profile feature extends further by facilitating the loading of parameters from both images and JSON files
> containing a profile. This means you can not only share your meticulously crafted settings but also import parameters
> from others, enabling collaborative exploration and knowledge exchange within the onnx-web community. As a
> user-friendly tool, onnx-web strives to enhance the customization and sharing aspects of image generation, providing
> users with a flexible and collaborative experience.

## Panorama pipeline

> onnx-web introduces a versatile panorama pipeline available for both SD v1.5 and SDXL, offering the flexibility to
> generate images without strict size limitations. Users can leverage this pipeline to produce images of 40 megapixels
> or larger, reaching dimensions such as 4k by 10k.
>
> The panorama pipeline operates by repeatedly running the original txt2img or img2img pipeline on numerous overlapping
> tiles within the image. A deliberate spiral pattern is employed, as it has demonstrated superior results compared to a
> grid arrangement. Depending on the chosen degree of tile overlap, users can achieve a completely seamless image.
> However, it's essential to note that increased overlap correlates with longer processing times, requiring a thoughtful
> balance based on user priorities.
>
> The panorama pipeline in onnx-web thus emerges as a sophisticated tool for users seeking expansive and detailed image
> generation. By combining the flexibility of the panorama approach with the precision afforded by region prompts and
> seeds, users can delve into intricate compositions and create visually captivating and seamless images.

### Region prompts

`<region:X:Y:W:H:S:F_TLBR:prompt+>`

> An intriguing feature within the panorama pipeline is the incorporation of region prompts and region seeds. Region
> prompts enable users to modify or replace the prompt within specific rectangular regions within the larger image. This
> functionality proves invaluable for adding characters to backgrounds, introducing cities into landscapes, or exerting
> precise control over where elements appear. It becomes a powerful tool for avoiding crowded or overlapping elements,
> offering nuanced control over image composition.

### Region seeds

`<reseed:X:Y:W:H:?:F_TLBR:seed>`

> Furthermore, region seeds enable the replication of the same object or image in different locations, even multiple
> times within the same image. To prevent hard edges and seamlessly integrate these region-based modifications, both
> region prompts and region seeds include options for blending with the surrounding prompt or seed. It's important to
> note that these region features are currently exclusive to the panorama pipeline.

## Grid mode

Makes many images. Takes many time.

> onnx-web introduces a powerful feature known as Grid Mode, designed to facilitate the efficient generation of multiple
> images with consistent parameters. Once enabled, Grid Mode allows users to select a parameter that varies across
> columns and another for rows. Users then provide specific values for each column or row, and the images are generated
> by combining the current parameters with the specified column and row values.
>
> In Grid Mode, selecting different parameters for columns and rows ensures diverse variations in the generated images.
> It's important to note that the same parameter cannot be chosen for both columns and rows unless the token replacement
> option is activated. Token replacement introduces the keywords column and row within the prompt, allowing users to
> dynamically insert the column or row values into their prompts before image generation.
>
> While there isn't a strict limit on the number of values users can provide for each dimension (columns and rows), it's
> essential to be mindful of the multiplicative effect. The more values provided, the longer the generation process will
> take. This trade-off allows users to balance their preferences for quantity and processing time based on their
> specific requirements.
>
> Grid Mode offers a versatile and time-efficient approach to exploring variations in image generation, making it a
> valuable tool for users seeking a range of outputs with nuanced parameter adjustments. By combining flexibility with
> precision, onnx-web empowers users to efficiently navigate the expansive creative possibilities within the Stable
> Diffusion process.

### Grid tokens

`__column__` and `__row__` if you pick token in the menu.

## Memory optimizations

> onnx-web introduces optimizations tailored for users with limited memory resources. The minimum system requirements
> are set at 4GB of VRAM and 8GB of system RAM for SD v1.5, escalating to 12GB of VRAM and 24GB of system RAM for SDXL.
> The recommended specifications stand at 6GB of VRAM and 16GB of system RAM for SD v1.5, increasing to 16GB of VRAM and
> 32GB of system RAM for SDXL.
>
> These optimizations cater to low-memory scenarios, providing users with the flexibility to adapt onnx-web to their
> hardware constraints while maintaining a balance between computational efficiency and image generation quality. Users
> can choose from these options based on their system specifications and performance preferences.

### Converting to fp16

Enable the option.

> The first optimization available is the integration of fp16 mode during model conversion. Employing fp16 significantly
> enhances efficiency in terms of storage space and inference, leading to faster runtime for fp16 models. While the
> results remain notably similar, it's imperative to acknowledge that not all models are compatible with fp16
> conversion, and caution is advised as it may disrupt certain Variational Autoencoders (VAEs).

### Moving models to the CPU

Option for each model.

> The second optimization caters to low-memory users by providing the option to offload specific models to the CPU.
> Notably, the UNet, being the largest model, is the primary candidate for GPU execution due to improved speed. However,
> onnx-web strategically offloads the text encoder and VAE to the CPU, recognizing that the text encoder only runs once
> at the beginning of each image, and the VAE operates once or twice at the image's outset and conclusion. This
> offloading approach proves especially impactful for SDXL, significantly mitigating memory constraints. While
> offloading the VAE might slightly affect high-resolution (highres) speed, it becomes a necessary trade-off to
> accommodate SDXL highres on certain GPUs with limited memory resources.
