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

export function getTheme(currentTheme: string, preferDark: boolean): string {
  if (currentTheme === '') {
    if (preferDark) {
      return 'dark';
    }
    return 'light';
  }
  return currentTheme;
}