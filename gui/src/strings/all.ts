import { I18N_STRINGS_DE } from './de.js';
import { I18N_STRINGS_EN } from './en.js';
import { I18N_STRINGS_ES } from './es.js';
import { I18N_STRINGS_FR } from './fr.js';

// easy way to make sure all locales have the complete set of strings
export type RequiredStrings = typeof I18N_STRINGS_EN['en']['translation'];

interface PartialLanguage {
  [key: string]: Omit<RequiredStrings, 'model' | 'platform' | 'scheduler'>;
}

export const I18N_STRINGS: Record<string, PartialLanguage> = {
  ...I18N_STRINGS_DE,
  ...I18N_STRINGS_EN,
  ...I18N_STRINGS_ES,
  ...I18N_STRINGS_FR,
};
