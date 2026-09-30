import test from 'node:test';
import assert from 'node:assert/strict';
import { describeProviders } from '../src/provider-config.js';

test('demo speech can independently use the real DeepSeek model', () => {
  assert.deepEqual(describeProviders({ asrProvider: 'demo', llmProvider: 'deepseek', demoMode: true, llmDemoMode: false }), {
    asrDemo: true, llmDemo: false, asrLabel: 'Текстовый источник', llmLabel: 'DeepSeek',
  });
});

test('a real speech server with mock extraction is not labeled real LLM', () => {
  const settings = describeProviders({ asrProvider: 'openai-compatible', llmProvider: 'demo', demoMode: false, llmDemoMode: true });
  assert.equal(settings.asrDemo, false);
  assert.equal(settings.llmDemo, true);
  assert.equal(settings.llmLabel, 'Демо без модели');
});

test('older configuration without llmDemoMode still distinguishes providers', () => {
  const settings = describeProviders({ asrProvider: 'demo', llmProvider: 'openai', demoMode: true });
  assert.equal(settings.asrDemo, true);
  assert.equal(settings.llmDemo, false);
  assert.equal(settings.llmLabel, 'OpenAI');
});

test('old all-demo configuration and unknown custom services remain readable', () => {
  assert.equal(describeProviders({ demoMode: true }).llmDemo, true);
  assert.equal(describeProviders({ asrProvider: 'speechmatics', llmProvider: 'openai-compatible' }).llmLabel, 'Свой сервер');
  assert.equal(describeProviders(undefined).asrDemo, false);
});
