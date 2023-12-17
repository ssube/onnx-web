# Getting Started

Welcome to onnx-web, a Stable Diffusion tool designed for straightforward and versatile use in AI art. Whether you're
running on an AMD or Nvidia GPU, on a Windows desktop or a Linux server, onnx-web is compatible with most machines.
It goes a step further by supporting multiple GPUs simultaneously, and with SDXL and LCM available for all platforms,
users can harness its capabilities without waiting. The panorama pipeline allows you to use regional prompts without
the need for additional plugins. Get ready to explore the basics of onnx-web and discover how it seamlessly fits into
your AI art toolkit.

The specialized features within onnx-web, such as the panorama pipeline and highres, are designed to produce
high-quality results in the megapixel scale from text prompts without human intervention or inpainting. The panorama
pipeline allows for the generation of large and seamless images, enhanced by the utilization of region prompts and
seeds. Highres, on the other hand, provides a super-resolution upscaling technique, refining images with iterative
img2img processes. Learn how to harness these features effectively, understanding the nuances of region-based
modifications, tokens, and optimal tile configurations.

Discover how onnx-web caters to users with varying hardware configurations. Uncover optimization techniques such as
converting models to fp16 mode, offloading computations to the CPU, and leveraging specialized features like the
panorama pipeline to further reduce memory usage. Gain insights into the considerations and trade-offs involved in
choosing parameters, tile configurations, and employing unique prompts for distinct creative outcomes.

This guide sets the stage for your onnx-web journey, offering a balance of technical depth and user-friendly insights to
empower you in your AI art exploration. Let's embark on this creative venture together, where innovation meets technical
precision.

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
  - [Adding your own models](#adding-your-own-models)
    - [Editing the extras file](#editing-the-extras-file)
  - [More details](#more-details)

## Setup

Before diving into the creative process, it's crucial to get onnx-web up and running on your system. We offer multiple
installation methods to cater to users of varying technical expertise. For beginners on Windows, an all-in-one EXE
bundle simplifies the installation process. Intermediate users can opt for a cross-platform installation using a Python
virtual environment, providing greater customization. Server administrators can explore OCI containers for streamlined
deployment. Keep in mind the minimum and recommended system requirements to ensure optimal performance, with options for
optimizations tailored to low-memory users.

### Windows bundle setup

For Windows users, the all-in-one EXE bundle is an accessible option. There are some minor limitations, for example you
cannot update dependencies or use the nightly ONNX packages, or use plugins that are not included in the bundle.

This bundle, distributed as a ZIP archive, requires extraction before execution. Make sure to disable the Windows
mark-of-the-web check to ensure the server runs successfully. Upon extraction, initiate onnx-web by running the EXE
file. Once model conversion is complete, the server will commence, opening a browser window for immediate access to the
web UI.

For more details, please see [the setup guide](./setup-guide.md#windows-all-in-one-bundle).

### Cross platform setup

Users who have a working Python environment and prefer installing their own dependencies can opt for a cross-platform
installation using a Python virtual environment. Ensure a functioning Python setup with either pip or conda. Begin by
cloning the onnx-web git repository, followed by installing the base requirements and those specific to your GPU
platform. Execute the launch script, patiently waiting for the model conversion process to conclude. Post-conversion,
open your browser and load the web UI for interaction.

For more details, please see [the setup guide](./setup-guide.md#cross-platform-method).

### Server setup with containers

For server administrators, onnx-web is also distributed as OCI containers, offering a streamlined deployment. Choose the
installation method that aligns with your proficiency and system requirements, whether it's the beginner-friendly EXE
bundle, the intermediate cross-platform setup, or the containerized deployment for server admins. Each pathway ensures a
straightforward onnx-web installation tailored to your technical needs.

For more details, please see [the server admin guide](./server-admin.md#containers).

## Running

Running onnx-web is the gateway to unlocking the creative potential of Stable Diffusion in AI art. Whether you are a
novice or an experienced enthusiast, this section guides you through the process of installing onnx-web on your system.

### Running the server

Initiate onnx-web by launching the Python server application, a process that demands your attention before proceeding
further. Specifically, ensure that your models are converted before the server commences its operation. For Windows
users, the distribution is available in the form of a convenient EXE bundle. Allow the server the necessary time to
perform this crucial model conversion, a prerequisite for optimal functionality during your creative sessions. Exercise
patience during this stage to guarantee a smooth experience with onnx-web.

### Running the web UI

After the server has successfully completed its startup procedures, access the web UI through your chosen evergreen
browser, including mobile browsers. Use the same URL and local port as the server to establish the essential connection.
This step solidifies the link between the server and the user interface, unlocking the full capabilities of onnx-web.
For those seeking additional functionality, explore ControlNet using your phone camera directly through the web UI. With
these meticulous steps completed, you are now equipped to efficiently harness the power of onnx-web and navigate the
realm of Stable Diffusion without compromise.

## Tabs

Understanding onnx-web's Core Features: onnx-web introduces a set of core features that form the backbone of your AI art
journey. The Stable Diffusion process, capable of running on both AMD and Nvidia GPUs, powers the image generation
pipeline. Explore the diverse tabs in the web UI, each offering unique functionalities such as text-based image
generation, upscaling, blending, and model management. Dive into the technical details of prompt syntax, model
conversions, and the intricacies of parameters, gaining a deeper understanding of how to fine-tune the AI art creation
process.

### Txt2img Tab

The txt2img tab in onnx-web serves the purpose of generating images from text prompts. Users can input textual
descriptions and witness the algorithm's creative interpretation, providing a seamless avenue for text-based image
generation.

For more details, please see [the user guide](./user-guide.md#txt2img-tab).

### Img2img Tab

For image-based prompts, the img2img tab is the go-to interface within onnx-web. Beyond its fundamental image
generation capabilities, this tab introduces the ControlNet mode, empowering users with advanced control over the
generated images through an innovative feature set.

For more details, please see [the user guide](./user-guide.md#img2img-tab).

### Inpaint Tab

The inpaint tab specializes in image generation with a unique combination of image prompts and masks. This
functionality allows users to guide the algorithm using both the source image and a mask, enhancing the precision and
customization of the generated content.

For more details, please see [the user guide](./user-guide.md#inpaint-tab).

### Upscale Tab

Addressing the need for higher resolutions, the upscale tab provides users with tools for high resolution and super
resolution. This feature is particularly useful for enhancing the quality and clarity of generated images, meeting
diverse artistic and practical requirements.

For more details, please see [the user guide](./user-guide.md#upscale-tab).

### Blend Tab

Enabling users to combine outputs or integrate external images, the blend tab in onnx-web offers a versatile blending
tool. This functionality adds a layer of creativity by allowing users to merge multiple outputs or incorporate
external images seamlessly.

For more details, please see [the user guide](./user-guide.md#blend-tab).

### Models Tab

Central to managing the core of onnx-web, the models tab provides users with the capability to manage Stable Diffusion
models. Additionally, it allows for the management of LoRAs (Latents of Random Ancestors) associated with these
models, facilitating a comprehensive approach to model customization.

For more details, please see [the user guide](./user-guide.md#models-tab).

### Settings Tab

Tailoring the user experience, the settings tab is the control center for managing onnx-web's web UI settings. Users
can configure server APIs, toggle dark mode for a personalized visual experience, and reset other tabs as needed,
ensuring a user-friendly and customizable environment.

For more details, please see [the user guide](./user-guide.md#settings-tab).

## Image parameters

In onnx-web, image parameters play a pivotal role in shaping the output of the Stable Diffusion process. These
parameters, including scheduler, CFG, steps, seed, batch size, prompt, optional negative prompt, and image width and
height, collectively govern the characteristics of the diffusion model's training and the resulting generated images.

### Common image parameters

These parameters are part of the Stable Diffusion pipeline and common to most tools.

- Scheduler
  - Role: The scheduler dictates the annealing schedule during the diffusion process.
  - Explanation: It determines how the noise level changes over time, influencing how the diffusion process resolves
    complex features like faces. Some schedulers are faster than others and some are more deterministic, reliably
    reproducing the same results.
  - LCM and Turbo require specific schedulers, which are marked in the web UI.
- Eta
  - Only applies to the DDIMScheduler, and is ignored in other schedulers.
  - A value of 0 corresponds to DDIM and 1 corresponds to DDPM.
- CFG
  - Role: CFG is integral for conditional image generation, allowing users to influence the generation based on specific
    conditions.
  - Explanation: By adjusting the CFG, users can guide the diffusion process to respond to certain prompts, achieving
    conditional outputs aligned with the specified criteria.
- Steps
  - Role: Steps determine the number of diffusion steps applied to the image.
  - Explanation: More steps generally result in a more refined image but require additional computation. Users can
    fine-tune this parameter based on the desired trade-off between quality and computational resources.
- Seed
  - Role: The seed initializes the randomization process, ensuring reproducibility.
  - Explanation: Setting a seed allows users to reproduce the same image by maintaining a consistent starting point for
    the random processes involved in the diffusion, facilitating result replication.
- Batch size
  - Role: Batch size influences the number of samples processed simultaneously.
  - Explanation: A larger batch size can expedite computation but may require more memory. It impacts the efficiency of
    the Stable Diffusion process, with users adjusting it based on available resources and computational preferences.
- Prompt
  - Role: The prompt provides the textual or visual input guiding the image generation.
  - Explanation: It serves as the creative input for the algorithm, shaping the direction of the generated content.
    Users articulate their artistic vision or preferences through carefully crafted prompts.
- Negative prompt
  - Role: Negative prompts offer a counterbalance to positive prompts, influencing the generation towards desired
    qualities or away from specific characteristics.
  - Explanation: By including a negative prompt, users can fine-tune the generated output, steering it away from
    undesired elements or towards a more nuanced and controlled result.
- Width, height
  - Role: These parameters define the dimensions of the generated image.
  - Explanation: Users specify the width and height to control the resolution and aspect ratio of the output. This
    allows for customization based on the intended use or artistic preferences.

### Unique image parameters

These parameters are unique to how onnx-web generates images.

- UNet tile size
  - One such parameter is the UNet tile size. This parameter governs the maximum size for each instance the UNet model
    runs. While it aids in reducing memory usage during panoramas and high-resolution processes, caution is needed.
    Reducing this below the image size in txt2img mode can result in repeated "totem pole" bodies, highlighting the
    importance of aligning the tile size appropriately with the intended use case.
- UNet overlap
  - The UNet overlap parameter plays a pivotal role in determining how much UNet tiles overlap. For most high-resolution
    applications, a value of 0.25 is recommended. However, for larger panoramas, opting for values between 0.5 to 0.75
    seamlessly blends tiles, significantly enhancing panorama quality. Balancing this parameter ensures optimal
    performance in diverse scenarios.
- Tiled VAE
  - For users engaging with the tiled VAE parameter, the choice revolves around whether the VAE (Variational
    Autoencoder) operates on the entire image or in smaller tiles. Opting for the tiled VAE not only accommodates larger
    images but also reduces VRAM usage. Notably, it doesn't exert a substantial impact on image quality, making it a
    pragmatic choice for scenarios where resource efficiency is a priority.
- VAE tile size
  - Parallel to UNet, the VAE (Variational Autoencoder) introduces two additional parameters: VAE tile size and VAE
    overlap. These mirror the UNet tile size and UNet overlap parameters, applying specifically to the VAE when the
    tiled VAE is active. Careful consideration of these parameters ensures effective utilization of onnx-web's
    capabilities while adapting to the unique requirements of your image generation tasks.
- VAE overlap
  - The amount of overlap between each VAE tile.

See the complete user guide for details about the highres, upscale, and correction parameters.

## Prompt syntax

Constructing prompts in onnx-web closely resembles familiar tools like auto1111, maintaining a consistent syntax
enclosed within angle brackets (`< >`). Notable differences arise, particularly in the handling of embeddings (Textual
Inversions) and regions. Despite these nuances, users transitioning from other tools will find the syntax comfortably
familiar, allowing for a seamless integration into onnx-web's creative environment.

### LoRAs and embeddings

`<lora:filename:1.0>` and `<embeddings:filename:1.0>` or `<inversion:filename:1.0>`.

You can include additional networks like LoRAs and embeddings (also known as Textual Inversions) using their respective
tokens. The syntax involves specifying the network type, name, and weight, separated by colons. For instance,
<lora:name:weight>. Most LoRA networks exhibit optimal performance within a strength range of 0.8 to 1.0. However, some
networks can be effectively utilized at higher or lower values, spanning from -1.0 to approximately 5.0 for certain
sliders. This flexibility provides users with nuanced control over the influence of LoRA networks on the generated
images.

For more details, please see [the user guide](./user-guide.md#lora-and-lycoris-tokens).

### CLIP skip

`<clip:skip:2>` for anime.

onnx-web supports CLIP skip through a token, allowing users to efficiently skip later stages in the CLIP keyword
hierarchy. This feature proves valuable, especially in genres like anime, where skipping specified levels can enhance
image results. For instance, skipping 2 levels refines the output by bypassing stages in the keyword hierarchy,
optimizing the creative outcome. This combination of tokens and functionalities enables users to precisely tailor
prompts in onnx-web for expressive image generation.

For more details, please see [the user guide](./user-guide.md#clip-skip-tokens).

## Highres

onnx-web supports a unique highres implementation, a powerful tool for super-resolution upscaling followed by img2img
refinement to restore intricate details. This feature can be iteratively applied to exponentially increase image
resolution. For instance, using a 2x upscaling model, two iterations of highres can produce an image four times the
original size, while three iterations result in an image eight times the size, and so forth.

The technique employed in highres draws parallels to the hires fix seen in other tools but uses repeated img2img runs,
similar to what SDXL does with the refiner. In this approach, the diffusion model refines its own output by running
img2img on smaller tiles, correcting the errors introduced by most super-resolution models. This technique allows
repeated iterative refinement, though it introduces a unique consideration – the img2img tiles do not perceive the
entire image context. Consequently, using a more generic prompt during the highres iterations can be will help avoid
recursive image features.

### Highres prompt

`txt2img prompt || img2img prompt`

Highres prompts are separated from the base txt2img prompt using the double pipe syntax (`||`). These prompts guide the
upscaling and refinement processes, enabling users to incorporate distinct instructions and achieve nuanced outputs.

For more details, please see [the user guide](./user-guide.md#prompt-stages).

### Highres iterations

`scale ** iterations`

Highres mode's iterative approach involves refining the generated image through multiple iterations, each contributing
to an exponential increase in resolution. Users can specify the number of iterations based on their desired level of
refinement. For instance, using a 2x upscaling model, two iterations of Highres will result in an image four times the
original size. This scaling effect continues exponentially with each additional iteration. The interplay of Highres
prompts and iterations allows users to progressively enhance image resolution while maintaining detailed and refined
visual elements.

For more details, please see [the user guide](./user-guide.md#highres-iterations-parameter).

## Profiles

The onnx-web web UI simplifies user experience with the introduction of a feature known as profiles. When you find
your favorite combination of image parameters, you can save them for future use as a named profiled. This is a great
way to reuse successful settings later and streamline your image generation process.

### Loading from files

Profiles can be loaded directly from image metadata or imported from JSON files shared by other users. Leveraging this
functionality, users can effortlessly reproduce specific parameters and settings that have been fine-tuned by their
peers. This collaborative approach not only streamlines the configuration process but also fosters a community-driven
exchange of preferences and techniques. Whether shared through image metadata or JSON files, loading profiles in
onnx-web offers users a convenient way to benefit from the collective insights and optimizations of the onnx-web
community.

### Saving profiles in the web UI

The process of saving profiles involves capturing the entire set of image parameters you've fine-tuned to achieve the
desired results. Once saved, these profiles are conveniently accessible for later usage. An additional benefit is the
ability to download your profiles as a JSON snippet. This JSON representation encapsulates the intricacies of your
selected parameters, allowing you to easily share your preferred configurations with others on platforms like Discord
or various online forums.

### Sharing parameters profiles

onnx-web's profile feature extends further by facilitating the loading of parameters from both images and JSON files
containing a profile. This means you can not only share your meticulously crafted settings but also import parameters
from others, enabling collaborative exploration and knowledge exchange within the onnx-web community. As a
user-friendly tool, onnx-web strives to enhance the customization and sharing aspects of image generation, providing
users with a flexible and collaborative experience.

TODO: link Discord profiles channel

## Panorama pipeline

onnx-web introduces a versatile panorama pipeline available for both SD v1.5 and SDXL, offering the flexibility to
generate images without strict size limitations. Users can leverage this pipeline to produce images of 40 megapixels
or larger, reaching dimensions such as 4k by 10k.

The panorama pipeline operates by repeatedly running the original txt2img or img2img pipeline on numerous overlapping
tiles within the image. A deliberate spiral pattern is employed, as it has demonstrated superior results compared to a
grid arrangement. Depending on the chosen degree of tile overlap, users can achieve a completely seamless image.
However, it's essential to note that increased overlap correlates with longer processing times, requiring a thoughtful
balance based on user priorities.

The panorama pipeline in onnx-web thus emerges as a sophisticated tool for users seeking expansive and detailed image
generation. By combining the flexibility of the panorama approach with the precision afforded by region prompts and
seeds, users can delve into intricate compositions and create visually captivating and seamless images.

### Region prompts

`<region:X:Y:W:H:S:F_TLBR:prompt+>`

An intriguing feature within the panorama pipeline is the incorporation of region prompts and region seeds. Region
prompts enable users to modify or replace the prompt within specific rectangular regions within the larger image. This
functionality proves invaluable for adding characters to backgrounds, introducing cities into landscapes, or exerting
precise control over where elements appear. It becomes a powerful tool for avoiding crowded or overlapping elements,
offering nuanced control over image composition.

For more details, please see [the user guide](./user-guide.md#region-tokens).

### Region seeds

`<reseed:X:Y:W:H:seed>`

Furthermore, region seeds enable the replication of the same object or image in different locations, even multiple
times within the same image. To prevent hard edges and seamlessly integrate these region-based modifications, both
region prompts and region seeds include options for blending with the surrounding prompt or seed. It's important to
note that these region features are currently exclusive to the panorama pipeline.

For more details, please see [the user guide](./user-guide.md#reseed-tokens-region-seeds).

## Grid mode

onnx-web introduces a powerful feature known as Grid Mode, designed to facilitate the efficient generation of multiple
images with consistent parameters. Once enabled, Grid Mode allows users to select a parameter that varies across
columns and another for rows. Users then provide specific values for each column or row, and the images are generated
by combining the current parameters with the specified column and row values.

In Grid Mode, selecting different parameters for columns and rows ensures diverse variations in the generated images.
It's important to note that the same parameter cannot be chosen for both columns and rows unless the token replacement
option is activated. Token replacement introduces the keywords column and row within the prompt, allowing users to
dynamically insert the column or row values into their prompts before image generation.

While there isn't a strict limit on the number of values users can provide for each dimension (columns and rows), it's
essential to be mindful of the multiplicative effect. The more values provided, the longer the generation process will
take. This trade-off allows users to balance their preferences for quantity and processing time based on their
specific requirements.

Grid Mode offers a versatile and time-efficient approach to exploring variations in image generation, making it a
valuable tool for users seeking a range of outputs with nuanced parameter adjustments. By combining flexibility with
precision, onnx-web empowers users to efficiently navigate the expansive creative possibilities within the Stable
Diffusion process.

### Grid tokens

`__column__` and `__row__`

When opting for token replacement, users can take advantage of the column and row tokens to dynamically insert column
and row values into their prompts. This feature is particularly powerful when working with comma-separated lists of
options. By utilizing these tokens, you gain the ability to precisely control how parameters evolve across the grid,
facilitating nuanced and controlled variations in image generation.

## Memory optimizations

onnx-web introduces optimizations tailored for users with limited memory resources. The system requirements are mostly
based on how much memory you have:

- minimum requirements
  - SD v1.5
    - 4GB VRAM
    - 8GB system RAM
  - SDXL
    - 12GB VRAM
    - 24GB system RAM
- recommended requirements
  - SD v1.5
    - 6GB VRAM
    - 16GB system RAM
  - SDXL
    - 16GB VRAM
    - 32GB system RAM

These optimizations cater to low-memory scenarios, providing users with the flexibility to adapt onnx-web to their
hardware constraints.

### Converting to fp16

The first optimization available is the integration of fp16 mode during model conversion. Employing fp16 significantly
enhances efficiency in terms of storage space and inference, leading to faster runtime for fp16 models. While the
results remain notably similar, it's imperative to acknowledge that not all models are compatible with fp16
conversion, and caution is advised as it may disrupt certain Variational Autoencoders (VAEs).

### Moving models to the CPU

The second optimization caters to low-memory users by providing the option to offload specific models to the CPU.
Notably, the UNet, being the largest model, is the primary candidate for GPU execution due to improved speed. However,
onnx-web strategically offloads the text encoder and VAE to the CPU, recognizing that the text encoder only runs once
at the beginning of each image, and the VAE operates once or twice at the image's outset and conclusion. This
offloading approach proves especially impactful for SDXL, significantly mitigating memory constraints. While
offloading the VAE might slightly affect high-resolution (highres) speed, it becomes a necessary trade-off to
accommodate SDXL highres on certain GPUs with limited memory resources.

## Adding your own models

onnx-web empowers users to seamlessly integrate their own models into the system through the utilization of an
extras.json file. This JSON file serves as a conduit for users to specify additional models, including LoRA networks and
embeddings, enhancing the versatility of onnx-web's capabilities.

Models that are in the models directory and follow the correct naming pattern will be shown in the web UI whether they
are listed in the extras file or not, but including them in the extras file allows you to provide a label for the web UI
and ensures that the hash is included in your output images.

### Editing the extras file

Begin by creating an extras.json file and defining your models within the designated structure. For instance:

```json
{
  "diffusion": [
    {
      "format": "safetensors",
      "name": "diffusion-sdxl-turbovision-v3-2",
      "label": "SDXL - Turbovision v3.2",
      "source": "civitai://255474?type=Model&format=SafeTensor&size=pruned&fp=fp16",
      "pipeline": "txt2img-sdxl"
    },
    {
      "format": "safetensors",
      "name": "diffusion-sdxl-icbinp-v1-0",
      "label": "SDXL - ICBINP",
      "source": "civitai://258447?type=Model&format=SafeTensor&size=pruned&fp=fp16",
      "pipeline": "txt2img-sdxl",
      "hash": "D6FF242DC70FC3DF8F311091FCD9A4DF3FDC1C85FEE2BCA604D4B8218A62378E"
    }
  ],
  "networks": [
    {
      "format": "safetensors",
      "label": "SDXL - LCM LoRA HuggingFace",
      "name": "sdxl-lcm",
      "source": "https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors",
      "tokens": [],
      "type": "lora"
    }
  ]
}
```

Save the extras.json file within the designated onnx-web directory. onnx-web will automatically download and convert the
specified models, streamlining the integration process without manual intervention.

Leverage the "label" field in your extras file to provide localized and user-friendly names for each model. These labels
will be seamlessly integrated into the onnx-web web UI, ensuring an intuitive interface for model selection.

## More details

For more details, please see the other guides:

- [API specification](./api.md)
- [custom chain pipelines](./chain-pipelines.md)
- [server administration](./server-admin.md)
- [setup guide](./setup-guide.md)
- [user guide](./user-guide.md)
