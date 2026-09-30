// Stateful linear resampling. Retains the boundary sample between render quanta.
export class PcmResampler {
  constructor(sourceRate, targetRate = 16000) {
    if (sourceRate <= 0 || targetRate <= 0) throw new Error('Invalid sample rate');
    this.ratio = sourceRate / targetRate;
    this.position = 0;
    this.total = 0;
    this.last = 0;
  }

  push(input) {
    if (!input.length) return new Int16Array(0);
    const end = this.total + input.length - 1;
    const result = [];
    while (this.position <= end) {
      const left = Math.floor(this.position);
      const fraction = this.position - left;
      if (fraction > 1e-8 && left + 1 > end) break;
      const a = left < this.total ? this.last : input[left - this.total];
      const b = fraction > 1e-8 ? input[left + 1 - this.total] : a;
      const sample = Math.max(-1, Math.min(1, a + (b - a) * fraction));
      result.push(Math.round(sample * (sample < 0 ? 32768 : 32767)));
      this.position += this.ratio;
    }
    this.last = input[input.length - 1];
    this.total += input.length;
    return Int16Array.from(result);
  }
}

export function packAudio(seq, sampleOffset, samples) {
  const packet = new ArrayBuffer(8 + samples.length * 2);
  const view = new DataView(packet);
  view.setUint32(0, seq, true);
  view.setUint32(4, sampleOffset, true);
  for (let index = 0; index < samples.length; index++) view.setInt16(8 + index * 2, samples[index], true);
  return packet;
}
