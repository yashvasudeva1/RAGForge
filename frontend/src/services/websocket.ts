import { ExecutionEvent } from '../types';

export class ExecutionWebSocketClient {
  private ws: WebSocket | null = null;
  private executionId: string;
  private onEventCallback: (event: ExecutionEvent) => void;
  private onErrorCallback?: (err: any) => void;

  constructor(
    executionId: string,
    onEvent: (event: ExecutionEvent) => void,
    onError?: (err: any) => void
  ) {
    this.executionId = executionId;
    this.onEventCallback = onEvent;
    this.onErrorCallback = onError;
  }

  connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const url = `${protocol}//${host}/api/executions/ws/${this.executionId}`;

    try {
      this.ws = new WebSocket(url);

      this.ws.onmessage = (message) => {
        try {
          const event: ExecutionEvent = JSON.parse(message.data);
          this.onEventCallback(event);
        } catch (e) {
          console.error('Failed to parse WS execution event', e);
        }
      };

      this.ws.onerror = (e) => {
        if (this.onErrorCallback) this.onErrorCallback(e);
      };

      this.ws.onclose = () => {
        this.ws = null;
      };
    } catch (e) {
      if (this.onErrorCallback) this.onErrorCallback(e);
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}
