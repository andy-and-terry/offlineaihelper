/**
 * Tests for ApiClient — mocks globalThis.fetch.
 */
import { jest } from '@jest/globals';
import { ApiClient, ServerUnavailableError, safeAsk } from '../src/client.js';

function mockFetch(status, body) {
  const response = {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? 'OK' : 'Error',
    json: async () => (typeof body === 'string' ? JSON.parse(body) : body),
  };
  globalThis.fetch = jest.fn().mockResolvedValue(response);
  return globalThis.fetch;
}

function mockFetchReject(error) {
  globalThis.fetch = jest.fn().mockRejectedValue(error);
  return globalThis.fetch;
}

afterEach(() => {
  if (globalThis.fetch && globalThis.fetch.mockRestore) {
    globalThis.fetch.mockRestore();
  }
  delete globalThis.fetch;
});

describe('ApiClient.ask', () => {
  test('returns result on success', async () => {
    mockFetch(200, { allowed: true, response: 'Paris', decision_code: 'ALLOW', reason: 'ok' });
    const client = new ApiClient({ baseUrl: 'http://localhost:11435' });
    const result = await client.ask('What is the capital of France?');
    expect(result.allowed).toBe(true);
    expect(result.response).toBe('Paris');
    expect(result.decision_code).toBe('ALLOW');
  });

  test('returns blocked result', async () => {
    mockFetch(200, { allowed: false, response: null, decision_code: 'BLOCK_DETERMINISTIC', reason: 'violence' });
    const client = new ApiClient({ baseUrl: 'http://localhost:11435' });
    const result = await client.ask('bad prompt');
    expect(result.allowed).toBe(false);
    expect(result.decision_code).toBe('BLOCK_DETERMINISTIC');
  });

  test('throws on 503', async () => {
    mockFetch(503, { detail: 'Ollama unavailable' });
    const client = new ApiClient({ baseUrl: 'http://localhost:11435' });
    await expect(client.ask('test')).rejects.toThrow('503');
  });
});

describe('ApiClient.moderate', () => {
  test('returns allowed decision', async () => {
    mockFetch(200, { allowed: true, decision_code: 'ALLOW', reason: 'ok' });
    const client = new ApiClient({ baseUrl: 'http://localhost:11435' });
    const result = await client.moderate('safe text', 'pre');
    expect(result.allowed).toBe(true);
    expect(result.decision_code).toBe('ALLOW');
  });

  test('returns blocked decision', async () => {
    mockFetch(200, { allowed: false, decision_code: 'BLOCK_LLM', reason: 'unsafe' });
    const client = new ApiClient({ baseUrl: 'http://localhost:11435' });
    const result = await client.moderate('dangerous text', 'pre');
    expect(result.allowed).toBe(false);
  });
});

describe('ApiClient.models', () => {
  test('returns available and configured models', async () => {
    mockFetch(200, {
      available: ['llama3.2:3b', 'llama-guard3:1b'],
      configured: { assistant: 'llama3.2:3b', moderator: 'llama-guard3:1b' },
    });
    const client = new ApiClient({ baseUrl: 'http://localhost:11435' });
    const result = await client.models();
    expect(result.available).toContain('llama3.2:3b');
    expect(result.configured.assistant).toBe('llama3.2:3b');
  });
});

describe('ApiClient.health', () => {
  test('returns ok status', async () => {
    mockFetch(200, { status: 'ok' });
    const client = new ApiClient({ baseUrl: 'http://localhost:11435' });
    const result = await client.health();
    expect(result.status).toBe('ok');
  });
});

describe('safeAsk', () => {
  test('returns result on success', async () => {
    mockFetch(200, { allowed: true, response: 'Hi!', decision_code: 'ALLOW', reason: 'ok' });
    const client = new ApiClient({ baseUrl: 'http://localhost:11435' });
    const result = await safeAsk(client, 'Hello');
    expect(result.allowed).toBe(true);
  });

  test('throws ServerUnavailableError on 503', async () => {
    mockFetch(503, { detail: 'server down' });
    const client = new ApiClient({ baseUrl: 'http://localhost:11435' });
    await expect(safeAsk(client, 'Hello')).rejects.toBeInstanceOf(ServerUnavailableError);
  });

  test('throws ServerUnavailableError on AbortError', async () => {
    const err = new Error('aborted');
    err.name = 'AbortError';
    mockFetchReject(err);
    const client = new ApiClient({ baseUrl: 'http://localhost:11435', timeoutMs: 100 });
    await expect(safeAsk(client, 'Hello')).rejects.toBeInstanceOf(ServerUnavailableError);
  });
});
