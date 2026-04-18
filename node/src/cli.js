#!/usr/bin/env node
/**
 * Node.js CLI for offlineaihelper.
 * Communicates with the Python API server over HTTP.
 */

import { program } from 'commander';
import chalk from 'chalk';
import { ApiClient, ServerUnavailableError } from './client.js';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const pkg = JSON.parse(readFileSync(join(__dirname, '../../package.json'), 'utf8'));

function makeClient(opts) {
  const baseUrl = opts.server ?? process.env.OFFLINEAIHELPER_SERVER_URL ?? 'http://127.0.0.1:11435';
  return new ApiClient({ baseUrl });
}

program
  .name('oah')
  .description('Offline AI assistant CLI (Node.js)')
  .version(pkg.version);

// ── ask ───────────────────────────────────────────────────────────────────
program
  .command('ask')
  .description('Send a prompt to the assistant')
  .requiredOption('-p, --prompt <text>', 'The prompt to send')
  .option('-s, --server <url>', 'Python API server URL')
  .action(async (opts) => {
    const client = makeClient(opts);
    try {
      const result = await client.ask(opts.prompt);
      if (result.allowed) {
        console.log(result.response);
      } else {
        console.error(chalk.yellow(`[BLOCKED] ${result.decision_code}: ${result.reason}`));
        process.exit(2);
      }
    } catch (err) {
      if (err instanceof ServerUnavailableError) {
        console.error(chalk.red(`Error: ${err.message}`));
      } else {
        console.error(chalk.red(`Unexpected error: ${err.message}`));
      }
      process.exit(1);
    }
  });

// ── moderate ──────────────────────────────────────────────────────────────
program
  .command('moderate')
  .description('Run text through the moderation pipeline')
  .requiredOption('-t, --text <text>', 'Text to moderate')
  .option('--stage <stage>', 'Moderation stage: pre or post', 'pre')
  .option('-s, --server <url>', 'Python API server URL')
  .action(async (opts) => {
    const client = makeClient(opts);
    try {
      const result = await client.moderate(opts.text, opts.stage);
      const color = result.allowed ? chalk.green : chalk.yellow;
      console.log(color(`Decision: ${result.decision_code}`));
      console.log(`Allowed : ${result.allowed}`);
      console.log(`Reason  : ${result.reason}`);
      if (!result.allowed) process.exit(2);
    } catch (err) {
      console.error(chalk.red(`Error: ${err.message}`));
      process.exit(1);
    }
  });

// ── check-models ──────────────────────────────────────────────────────────
program
  .command('check-models')
  .description('Show available and configured Ollama models')
  .option('-s, --server <url>', 'Python API server URL')
  .action(async (opts) => {
    const client = makeClient(opts);
    try {
      const data = await client.models();
      console.log(chalk.bold('\nAvailable models:'));
      for (const m of data.available) console.log(`  • ${m}`);
      console.log(chalk.bold('\nConfigured models:'));
      for (const [alias, model] of Object.entries(data.configured)) {
        const present = data.available.some((m) => m === model || m.startsWith(model));
        const status = present ? chalk.green('✓ present') : chalk.red('✗ missing');
        console.log(`  [${alias}] ${model}  ${status}`);
      }
    } catch (err) {
      console.error(chalk.red(`Error: ${err.message}`));
      process.exit(1);
    }
  });

// ── health ────────────────────────────────────────────────────────────────
program
  .command('health')
  .description('Check if the Python API server is running')
  .option('-s, --server <url>', 'Python API server URL')
  .action(async (opts) => {
    const client = makeClient(opts);
    try {
      const result = await client.health();
      console.log(chalk.green(`Server is ${result.status}`));
    } catch (err) {
      console.error(chalk.red(`Server not reachable: ${err.message}`));
      process.exit(1);
    }
  });

program.parse();
