export function imageFromBlob(blob: Blob): Promise<HTMLImageElement> {
  return new Promise((res, rej) => {
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(src);
      res(image);
    };

    const src = URL.createObjectURL(blob);
    image.src = src;
  });
}
