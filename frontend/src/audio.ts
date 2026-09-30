import { packAudio } from '../public/audio-core.js';
import type { Snapshot } from './types';

const AUDIO = { encoding: 'pcm_s16le', sampleRate: 16000, channels: 1 };
type ConnectionState = 'connecting' | 'connected' | 'reconnecting';
interface Callbacks {
  snapshot: (snapshot: Snapshot) => void;
  partial: (text: string) => void;
  connection: (state: ConnectionState) => void;
  error: (message: string) => void;
  level: (value: number) => void;
  microphone: (active: boolean) => void;
}

class MicrophoneCapture {
  private flushResolver?: () => void;
  private stopTask?: Promise<void>;
  private constructor(
    private context: AudioContext,
    private media: MediaStream,
    private source: MediaStreamAudioSourceNode,
    private worklet: AudioWorkletNode,
    private silent: GainNode,
  ) {}

  static async create(onSamples: (samples: Int16Array) => void) {
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext) {
      throw new Error('Микрофон доступен на localhost или через HTTPS в современном браузере.');
    }
    const context = new AudioContext();
    await context.resume();
    let media: MediaStream | undefined;
    try {
      media = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }, video: false });
      await context.audioWorklet.addModule('/pcm-capture.worklet.js');
      const source = context.createMediaStreamSource(media);
      const worklet = new AudioWorkletNode(context, 'pcm-capture', { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1 });
      const silent = context.createGain();
      silent.gain.value = 0;
      const capture = new MicrophoneCapture(context, media, source, worklet, silent);
      worklet.port.onmessage = ({ data }: MessageEvent<{ type: string; buffer: ArrayBuffer }>) => {
        if (data.type === 'samples') onSamples(new Int16Array(data.buffer));
        if (data.type === 'flushed') capture.flushResolver?.();
      };
      return capture;
    } catch (error) {
      media?.getTracks().forEach(track => track.stop());
      await context.close();
      throw error;
    }
  }

  start() {
    this.source.connect(this.worklet);
    this.worklet.connect(this.silent);
    this.silent.connect(this.context.destination);
  }

  stop(): Promise<void> {
    this.stopTask ??= this.flushAndClose();
    return this.stopTask;
  }

  private async flushAndClose() {
    if (this.context.state === 'closed') return;
    try {
      await new Promise<void>((resolve, reject) => {
        const timer = window.setTimeout(() => reject(new Error('Не удалось получить последний фрагмент звука из браузера. Микрофон остановлен; повторно завершите запись для обработки уже переданного аудио.')), 3000);
        this.flushResolver = () => { window.clearTimeout(timer); resolve(); };
        this.worklet.port.postMessage('flush');
      });
    } finally {
      this.source.disconnect();
      this.worklet.disconnect();
      this.silent.disconnect();
      this.media.getTracks().forEach(track => track.stop());
      await this.context.close();
    }
  }
}

export class SessionConnection {
  private socket?: WebSocket;
  private closed = false;
  private timer?: number;
  private reconnectAttempt = 0;
  private capture?: MicrophoneCapture;
  private streamId?: string;
  private nextSeq = 1;
  private sampleOffset = 0;
  private queue = new Map<number, ArrayBuffer>();
  private bufferedSamples = 0;
  private streaming = false;
  private streamReady = false;
  private overflowed = false;
  private pendingStart?: { resolve: () => void; reject: (error: Error) => void };
  private ackWaiters: Array<() => void> = [];

  constructor(private sessionId: string, private callbacks: Callbacks) { this.connect(); }

  private connect() {
    if (this.closed) return;
    this.streamReady = false;
    this.callbacks.connection(this.reconnectAttempt ? 'reconnecting' : 'connecting');
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${protocol}//${location.host}/api/v1/sessions/${this.sessionId}/stream`);
    this.socket = socket;
    socket.onopen = () => {
      if (this.closed || this.socket !== socket) return;
      this.reconnectAttempt = 0;
      this.callbacks.connection('connected');
      if (this.streamId && (this.streaming || this.queue.size)) this.send({ type: 'stream.resume', streamId: this.streamId, audio: AUDIO });
      else if (this.pendingStart) this.send({ type: 'stream.start', audio: AUDIO });
    };
    socket.onmessage = ({ data }) => {
      if (this.closed || this.socket !== socket || typeof data !== 'string') return;
      try {
        const event = JSON.parse(data);
        const payload = event.payload ?? {};
        if (event.type === 'session.snapshot') {
          this.callbacks.snapshot(payload as Snapshot);
          if (payload.status === 'error' || payload.status === 'stopped') {
            this.streamReady = false;
            this.streaming = false;
            this.overflowed = true;
            void this.capture?.stop().catch(error => this.callbacks.error(String(error)));
            this.capture = undefined;
            this.callbacks.microphone(false);
            this.callbacks.level(0);
            this.pendingStart?.reject(new Error(payload.error || 'Ошибка обработки записи.'));
            this.pendingStart = undefined;
          }
        }
        if (event.type === 'transcript.partial') this.callbacks.partial(payload.text || '');
        if (event.type === 'stream.ready') {
          this.streamId = payload.streamId;
          this.acknowledge(payload.throughSeq || 0);
          this.nextSeq = Math.max(this.nextSeq, (payload.throughSeq || 0) + 1);
          this.sampleOffset = Math.max(this.sampleOffset, payload.nextSampleOffset || 0);
          for (const packet of this.queue.values()) socket.send(packet);
          this.streamReady = true;
          this.pendingStart?.resolve();
          this.pendingStart = undefined;
        }
        if (event.type === 'audio.ack' && payload.streamId === this.streamId) this.acknowledge(payload.throughSeq);
        if (event.type === 'error') {
          const message = payload.message || payload.detail || 'Не удалось обработать поток. Попробуйте новую консультацию.';
          this.callbacks.error(message);
          this.pendingStart?.reject(new Error(message));
          this.pendingStart = undefined;
        }
      } catch {
        this.callbacks.error('Сервер прислал некорректное событие. Обновите страницу.');
      }
    };
    socket.onclose = event => {
      if (this.closed || this.socket !== socket) return;
      this.streamReady = false;
      this.callbacks.connection('reconnecting');
      if (event.code === 1008) {
        this.callbacks.error('Подключение к консультации отклонено. Возможно, она уже открыта в другой вкладке. Закройте вторую вкладку и обновите страницу.');
        this.streaming = false;
        this.overflowed = true;
        void this.capture?.stop().catch(error => this.callbacks.error(String(error)));
        this.capture = undefined;
        this.callbacks.microphone(false);
        this.pendingStart?.reject(new Error('Подключение отклонено сервером.'));
        this.pendingStart = undefined;
        return;
      }
      this.timer = window.setTimeout(() => this.connect(), Math.min(5000, 500 * 2 ** this.reconnectAttempt++));
    };
    socket.onerror = () => socket.close();
  }

  private send(event: object) {
    if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(JSON.stringify(event));
  }

  private acknowledge(throughSeq: number) {
    for (const [seq, packet] of this.queue) {
      if (seq <= throughSeq) { this.bufferedSamples -= (packet.byteLength - 8) / 2; this.queue.delete(seq); }
    }
    if (!this.queue.size) this.ackWaiters.splice(0).forEach(resolve => resolve());
  }

  private onSamples = (samples: Int16Array) => {
    if (this.closed || this.overflowed || !samples.length) return;
    if (this.bufferedSamples + samples.length > 30 * 16000) {
      this.overflowed = true;
      void this.capture?.stop().catch(error => this.callbacks.error(String(error)));
      this.capture = undefined;
      this.callbacks.microphone(false);
      this.callbacks.level(0);
      this.callbacks.error('Связь с сервером прервана больше чем на 30 секунд. Микрофон остановлен. Восстановите связь и завершите запись, чтобы обработать сохранённую часть.');
      return;
    }
    let squares = 0;
    for (const sample of samples) squares += (sample / 32768) ** 2;
    this.callbacks.level(Math.min(1, Math.sqrt(squares / samples.length) * 5));
    const packet = packAudio(this.nextSeq, this.sampleOffset, samples);
    this.queue.set(this.nextSeq++, packet);
    this.sampleOffset += samples.length;
    this.bufferedSamples += samples.length;
    if (this.streamReady && this.socket?.readyState === WebSocket.OPEN && this.socket.bufferedAmount < 512 * 1024) this.socket.send(packet);
    else if (this.streamReady && this.socket?.readyState === WebSocket.OPEN) this.socket.close();
  };

  async startMicrophone() {
    if (this.capture || this.streaming) return;
    this.streamReady = false;
    this.overflowed = false;
    this.capture = await MicrophoneCapture.create(this.onSamples);
    if (this.closed) { await this.capture.stop(); return; }
    try {
      await new Promise<void>((resolve, reject) => {
        const timer = window.setTimeout(() => {
          this.pendingStart = undefined;
          reject(new Error('Сервер не подтвердил начало записи. Проверьте подключение к распознаванию речи.'));
        }, 35000);
        this.pendingStart = {
          resolve: () => { window.clearTimeout(timer); resolve(); },
          reject: error => { window.clearTimeout(timer); reject(error); },
        };
        this.send({ type: 'stream.start', audio: AUDIO });
      });
      if (this.closed) return;
      this.streaming = true;
      this.capture.start();
      this.callbacks.microphone(true);
    } catch (error) {
      await this.capture?.stop().catch(() => undefined);
      this.capture = undefined;
      throw error;
    }
  }

  async finishAudio() {
    try { await this.capture?.stop(); }
    finally {
      this.capture = undefined;
      this.callbacks.microphone(false);
      this.callbacks.level(0);
    }
    if (this.queue.size) {
      await new Promise<void>((resolve, reject) => {
        const timer = window.setTimeout(() => reject(new Error('Ожидаем доставку аудио на сервер. После восстановления связи нажмите «Завершить» ещё раз.')), 15000);
        this.ackWaiters.push(() => { window.clearTimeout(timer); resolve(); });
      });
    }
    this.streaming = false;
  }

  close() {
    this.closed = true;
    if (this.timer) window.clearTimeout(this.timer);
    this.pendingStart?.reject(new Error('Консультация закрыта.'));
    this.pendingStart = undefined;
    void this.capture?.stop().catch(() => undefined);
    this.capture = undefined;
    this.socket?.close();
  }
}
