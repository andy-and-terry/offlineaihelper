/**
 * Tests for CLI helper logic — verifies client method calls and exit codes.
 */
import { jest } from '@jest/globals';
import { ApiClient } from '../src/client.js';

describe('ApiClient integration scenarios', () => {
  test('ask with allowed response', async () => {
    globalThis.fetch = jest.fn().mockResolvedValue({
      ok: true, status: 200,
      json: async () => ({ allowed: true, response: 'The answer is 42', decision_code: 'ALLOW', reason: 'ok' }),
    });
    const client = new ApiClient({ baseUrl: 'http://localhost:11435' });
    const result = await client.ask('What is the meaning of life?');
    expect(result.allowed).toBe(true);
    expect(result.response).toBe('The answer is 42');
    delete globalThis.fetch;
  });

  test('ask with blocked response', async () => {
    globalThis.fetch = jest.fn().mockResolvedValue({
      ok: true, status: 200,
      json: async () => ({ allowed: false, response: null, decision_code: 'BLOCK_DETERMINISTIC', reason: 'violence' }),
    });
    const client = new ApiClient({ baseUrl: 'http://localhost:11435' });
    const result = await client.ask('harmful prompt');
    expect(result.allowed).toBe(false);
    expect(result.decision_code).toBe('BLOCK_DETERMINISTIC');
    delete globalThis.fetch;
  });

  test('moderate returns correct decision', async () => {
    globalThis.fetch = jest.fn().mockResolvedValue({
      ok: true, status: 200,
      json: async () => ({ allowed: true, decision_code: 'ALLOW', reason: 'ok' }),
    });
    const client = new ApiClient({ baseUrl: 'http://localhost:11435' });
    const result = await client.moderate('safe text', 'pre');
    expect(result.decision_code).toBe('ALLOW');
    delete globalThis.fetch;
  });

  test('check-models returns model lists', async () => {
    globalThis.fetch = jest.fn().mockResolvedValue({
      ok: true, status: 200,
      json: async () => ({
        available: ['llama3.2:3b', 'llama-guard3:1b'],
        configured: { assistant: 'llama3.2:3b', moderator: 'llama-guard3:1b' },
      }),
    });
    const client = new ApiClient({ baseUrl: 'http://localhost:11435' });
    const data = await client.models();
    expect(data.available.length).toBe(2);
    expect(data.configured).toHaveProperty('assistant');
    delete globalThis.fetch;
  });

  test('health check returns ok', async () => {
    globalThis.fetch = jest.fn().mockResolvedValue({
      ok: true, status: 200,
      json: async () => ({ status: 'ok' }),
    });
    const client = new ApiClient({ baseUrl: 'http://localhost:11435' });
    const result = await client.health();
    expect(result.status).toBe('ok');
    delete globalThis.fetch;
  });
});
