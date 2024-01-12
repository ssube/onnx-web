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

export function trimHash(val: string): string {
  if (val[0] === '#') {
    return val.slice(1);
  }

  return val;
}

/**
 * from https://stackoverflow.com/a/30800715
 */
export function downloadAsJson(data: object, filename = 'parameters.json'): void {
  const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(data));
  const elem = document.createElement('a');
  elem.setAttribute('href', dataStr);
  elem.setAttribute('download', filename);
  document.body.appendChild(elem); // required for firefox
  elem.click();
  elem.remove();
}
