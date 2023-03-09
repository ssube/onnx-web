import { BaseError } from 'noicejs';
import { ApiClient } from './api.js';

export class NoServerError extends BaseError {
  constructor() {
    super('cannot connect to server');
  }
}

/**
 * @TODO client-side inference with https://www.npmjs.com/package/onnxruntime-web
 */
export const LOCAL_CLIENT = {
  async masks() {
    throw new NoServerError();
  },
  async blend(model, params, upscale) {
    throw new NoServerError();
  },
  async img2img(model, params, upscale) {
    throw new NoServerError();
  },
  async txt2img(model, params, upscale) {
    throw new NoServerError();
  },
  async inpaint(model, params, upscale) {
    throw new NoServerError();
  },
  async upscale(model, params, upscale) {
    throw new NoServerError();
  },
  async outpaint(model, params, upscale) {
    throw new NoServerError();
  },
  async noises() {
    throw new NoServerError();
  },
  async params() {
    throw new NoServerError();
  },
  async ready(key) {
    throw new NoServerError();
  },
  async cancel(key) {
    throw new NoServerError();
  },
  async models() {
    throw new NoServerError();
  },
  async platforms() {
    throw new NoServerError();
  },
  async schedulers() {
    throw new NoServerError();
  },
  async strings() {
    return {};
  },
} as ApiClient;
