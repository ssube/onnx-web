export const I18N_STRINGS_EN = {
  en: {
    translation: {
      history: {
        empty: 'No results. Press Generate to create an image.',
      },
      loading: {
        cancel: 'Cancel',
        progress: '{{current}} of {{total}} steps',
        unknown: 'many',
      },
      tab: {
        blend: 'Blend',
        img2img: 'Img2img',
        inpaint: 'Inpaint',
        txt2txt: 'Txt2txt',
        txt2img: 'Txt2img',
        upscale: 'Upscale',
      },
      tooltip: {
        delete: 'Delete',
        next: 'EN Next',
        previous: 'EN Previous',
        save: 'Save',
      },
    }
  },
};

// easy way to make sure all locales have the complete set of strings
export type RequiredStrings = typeof I18N_STRINGS_EN['en'];
