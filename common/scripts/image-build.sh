#! /bin/sh

IMAGE_PUSH="${1:---skip}"

IMAGE_NAME="${CI_PROJECT_PATH}-${IMAGE_SUFFIX}"
IMAGE_TAG="${IMAGE_ARCH}-$(echo "${CI_COMMIT_TAG:-${CI_COMMIT_REF_SLUG}}" | sed -r 's/[^-_a-zA-Z0-9\\.]/-/g')"

IMAGE_FULL="${IMAGE_NAME}:${IMAGE_TAG}"

echo "Loading last image: ${IMAGE_FULL}"

docker pull "${IMAGE_FULL}" || echo "Failed to load last image."

echo "Building image: ${IMAGE_FULL}"

docker build -f "${IMAGE_FILE}" -t "${IMAGE_FULL}" . || { echo "Failed to build image!"; exit 1; }

if [[ "${IMAGE_PUSH}" == "--push" ]];
then
  echo "Pushing image: ${IMAGE_FULL}"
  docker push "${IMAGE_FULL}" || { echo "Failed to push image!"; exit 1; }
fi
