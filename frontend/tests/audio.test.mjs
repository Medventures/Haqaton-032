import test from 'node:test';
import assert from 'node:assert/strict';
import { PcmResampler, packAudio } from '../public/audio-core.js';

test('48 kHz capture produces precisely 16 kHz PCM across 128-sample worklet quanta', () => {
  const resampler = new PcmResampler(48000);
  const input = Float32Array.from({ length: 48000 }, (_, i) => Math.sin(i / 100));
  const parts = [];
  for (let i = 0; i < input.length; i += 128) parts.push(...resampler.push(input.subarray(i, i + 128)));
  assert.equal(parts.length, 16000);
  assert.deepEqual(Int16Array.from(parts), new PcmResampler(48000).push(input));
});

test('fractional 44.1 kHz resampling keeps interpolation continuous at chunk boundaries', () => {
  const input = Float32Array.from({ length: 44100 }, (_, i) => Math.sin(i / 51));
  const resampler = new PcmResampler(44100);
  const parts = [];
  for (let i = 0; i < input.length; i += 128) parts.push(...resampler.push(input.subarray(i, i + 128)));
  assert.equal(parts.length, 16000);
  assert.deepEqual(Int16Array.from(parts), new PcmResampler(44100).push(input));
});

test('PCM frame uses agreed little endian sequence, offset and signed samples', () => {
  const packed = packAudio(260, 3200, new Int16Array([-32768, -1, 0, 32767]));
  assert.deepEqual([...new Uint8Array(packed)], [4, 1, 0, 0, 128, 12, 0, 0, 0, 128, 255, 255, 0, 0, 255, 127]);
});

test('resampler clips values and handles empty input', () => {
  const resampler = new PcmResampler(16000);
  assert.equal(resampler.push(new Float32Array()).length, 0);
  assert.deepEqual([...resampler.push(new Float32Array([-2, 2, 0]))], [-32768, 32767, 0]);
});
