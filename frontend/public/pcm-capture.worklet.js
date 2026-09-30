import { PcmResampler } from './audio-core.js';

class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.resampler = new PcmResampler(sampleRate, 16000);
    this.pending = [];
    this.active = true;
    this.port.onmessage = ({ data }) => {
      if (data === 'flush') {
        this.active = false;
        this.emit();
        this.port.postMessage({ type: 'flushed' });
      }
    };
  }
  emit() {
    if (!this.pending.length) return;
    const samples = Int16Array.from(this.pending);
    this.pending = [];
    this.port.postMessage({ type: 'samples', buffer: samples.buffer }, [samples.buffer]);
  }
  process(inputs) {
    if (!this.active) return true;
    const input = inputs[0]?.[0];
    if (input) {
      const samples = this.resampler.push(input);
      for (const sample of samples) {
        this.pending.push(sample);
        if (this.pending.length >= 3200) this.emit();
      }
    }
    return true;
  }
}
registerProcessor('pcm-capture', PcmCapture);
