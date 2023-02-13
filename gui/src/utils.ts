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

export function range(max: number): Array<number> {
  return [...Array(max).keys()];
}

export function visibleIndex(idx: number): string {
  return (idx + 1).toFixed(0);
}
