/** @param {{ asrProvider?: string, llmProvider?: string, demoMode?: boolean, llmDemoMode?: boolean } | undefined} config */
export function describeProviders(config) {
  const asrDemo = config?.asrProvider ? config.asrProvider === 'demo' : !!config?.demoMode;
  const llmDemo = config?.llmDemoMode ?? (config?.llmProvider ? config.llmProvider === 'demo' : !!config?.demoMode);
  const llmNames = { deepseek: 'DeepSeek', openai: 'OpenAI', 'openai-compatible': 'Свой сервер' };
  const asrNames = { speechmatics: 'Speechmatics', 'openai-compatible': 'Свой сервер распознавания' };
  return {
    asrDemo,
    llmDemo,
    llmLabel: llmDemo ? 'Демо без модели' : llmNames[config?.llmProvider] || config?.llmProvider || 'Подключаемся',
    asrLabel: asrDemo ? 'Текстовый источник' : asrNames[config?.asrProvider] || config?.asrProvider || 'Микрофон',
  };
}
