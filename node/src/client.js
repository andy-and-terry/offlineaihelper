/**
 * HTTP client for the offlineaihelper Python API server.
 * Uses the built-in globalThis.fetch available in Node >= 18.
 */

export class ApiClient {
  /**
   * @param {object} opts
   * @param {string} [opts.baseUrl] - Base URL of the Python server
   * @param {number} [opts.timeoutMs] - Request timeout in milliseconds
   */
  constructor({ baseUrl = 'http://127.0.0.1:11435', timeoutMs = 30_000 } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.timeoutMs = timeoutMs;
  }

  /** @private */
  async _fetch(path, init = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const res = await fetch(`${this.baseUrl}${path}`, {
        ...init,
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json', ...(init.headers ?? {}) },
      });
      if (!res.ok) {
        let detail = res.statusText;
        try { detail = (await res.json()).detail ?? detail; } catch {}
        const err = new Error(`API error ${res.status}: ${detail}`);
        err.statusCode = res.status;
        throw err;
      }
      return res.json();
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * Send a prompt to the assistant.
   * @param {string} prompt
   * @returns {Promise<{allowed:boolean, response:string|null, decision_code:string, reason:string}>}
   */
  async ask(prompt) {
    return this._fetch('/ask', { method: 'POST', body: JSON.stringify({ prompt }) });
  }

  /**
   * Run moderation on a piece of text.
   * @param {string} text
   * @param {'pre'|'post'} [stage='pre']
   * @returns {Promise<{allowed:boolean, decision_code:string, reason:string}>}
   */
  async moderate(text, stage = 'pre') {
    return this._fetch('/moderate', { method: 'POST', body: JSON.stringify({ text, stage }) });
  }

  /**
   * Get available and configured models.
   * @returns {Promise<{available:string[], configured:Record<string,string>}>}
   */
  async models() {
    return this._fetch('/models');
  }

  /**
   * Health-check the Python server.
   * @returns {Promise<{status:string}>}
   */
  async health() {
    return this._fetch('/health');
  }
}

export class ServerUnavailableError extends Error {
  constructor(message) {
    super(message);
    this.name = 'ServerUnavailableError';
  }
}

/**
 * Wraps ApiClient.ask() with server-availability error handling.
 * @param {ApiClient} client
 * @param {string} prompt
 */
export async function safeAsk(client, prompt) {
  try {
    return await client.ask(prompt);
  } catch (err) {
    if (err.name === 'AbortError' || err.statusCode === 503) {
      throw new ServerUnavailableError(
        'Python API server is not reachable. Start it with: offlineaihelper serve'
      );
    }
    throw err;
  }
}
