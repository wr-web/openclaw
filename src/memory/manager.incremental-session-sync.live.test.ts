/**
 * Live integration test for incremental session sync with a real Ollama
 * embedding provider (nomic-embed-text:latest, 768 dims).
 *
 * Run with:
 *   OLLAMA_BASE_URL=http://localhost:11434 pnpm test:live \
 *     -- src/memory/manager.incremental-session-sync.live.test.ts
 *
 * Skipped automatically when Ollama is unreachable.
 */

import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";
import type { OpenClawConfig } from "../config/config.js";

const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL ?? "http://localhost:11434";
const OLLAMA_MODEL = "nomic-embed-text:latest";

// ---------------------------------------------------------------------------
// Skip the whole suite if Ollama is not reachable
// ---------------------------------------------------------------------------
let ollamaReachable = false;
beforeAll(async () => {
  try {
    const res = await fetch(`${OLLAMA_BASE_URL}/api/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: OLLAMA_MODEL, prompt: "ping" }),
      signal: AbortSignal.timeout(5000),
    });
    ollamaReachable = res.ok;
  } catch {
    ollamaReachable = false;
  }
  if (!ollamaReachable) {
    console.warn(`[live] Ollama not reachable at ${OLLAMA_BASE_URL} — skipping suite`);
  }
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeSessionLine(role: "user" | "assistant", content: string): string {
  return JSON.stringify({ type: "message", message: { role, content } });
}

function buildTranscript(count: number): string {
  const lines: string[] = [];
  for (let i = 0; i < count; i++) {
    const role = i % 2 === 0 ? "user" : "assistant";
    lines.push(makeSessionLine(role, `Message ${i + 1}: ${"x".repeat(40)}`));
  }
  return lines.join("\n") + "\n";
}

function createCfg(params: {
  workspaceDir: string;
  storePath: string;
  stateDir: string;
}): OpenClawConfig {
  return {
    agents: {
      defaults: {
        workspace: params.workspaceDir,
        memorySearch: {
          provider: "ollama",
          model: OLLAMA_MODEL,
          remote: {
            baseUrl: OLLAMA_BASE_URL,
            apiKey: "ollama-local",
          },
          store: { path: params.storePath, vector: { enabled: false } },
          chunking: { tokens: 300, overlap: 0 },
          sync: { watch: false, onSessionStart: false, onSearch: false },
          query: { minScore: 0, hybrid: { enabled: false } },
          sources: ["sessions"],
          experimental: { sessionMemory: true },
        },
      },
      list: [{ id: "main", default: true }],
    },
  } as unknown as OpenClawConfig;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("incremental session sync — live Ollama (nomic-embed-text)", () => {
  let tmpRoot: string;
  let workspaceDir: string;
  let stateDir: string;
  let sessionDir: string;
  let sessionFile: string;
  let storePath: string;
  let previousStateDir: string | undefined;

  // Lazy-import so module-level mocks in other test files don't interfere.
  let getMemorySearchManager: (typeof import("./index.js"))["getMemorySearchManager"];
  let closeAllMemorySearchManagers: (typeof import("./index.js"))["closeAllMemorySearchManagers"];

  beforeAll(async () => {
    ({ getMemorySearchManager, closeAllMemorySearchManagers } = await import("./index.js"));
  });

  beforeEach(async () => {
    tmpRoot = await fs.mkdtemp(path.join(os.tmpdir(), "openclaw-inc-live-"));
    workspaceDir = path.join(tmpRoot, "workspace");
    stateDir = path.join(tmpRoot, "state");
    sessionDir = path.join(stateDir, "agents", "main", "sessions");
    sessionFile = path.join(sessionDir, "live-convo.jsonl");
    storePath = path.join(tmpRoot, "index.sqlite");

    await fs.mkdir(path.join(workspaceDir, "memory"), { recursive: true });
    await fs.mkdir(sessionDir, { recursive: true });

    previousStateDir = process.env.OPENCLAW_STATE_DIR;
    process.env.OPENCLAW_STATE_DIR = stateDir;
  });

  afterEach(async () => {
    await closeAllMemorySearchManagers();
    if (previousStateDir === undefined) {
      delete process.env.OPENCLAW_STATE_DIR;
    } else {
      process.env.OPENCLAW_STATE_DIR = previousStateDir;
    }
    await fs.rm(tmpRoot, { recursive: true, force: true });
  });

  it("only calls Ollama for new messages on incremental sync", async () => {
    if (!ollamaReachable) {
      return; // skip
    }

    // -----------------------------------------------------------------------
    // Phase 1 — initial sync of 30 messages
    // -----------------------------------------------------------------------
    const INITIAL = 30;
    await fs.writeFile(sessionFile, buildTranscript(INITIAL));

    const result = await getMemorySearchManager({
      cfg: createCfg({ workspaceDir, storePath, stateDir }),
      agentId: "main",
    });
    const manager = result.manager!;

    const t0 = performance.now();
    await manager.sync({ reason: "live-initial" });
    const t1 = performance.now();

    const statusAfterInitial = manager.status();
    const chunksAfterInitial = statusAfterInitial.chunks;

    console.log(`[live] Initial sync: ${chunksAfterInitial} chunks, ${(t1 - t0).toFixed(0)} ms`);
    expect(chunksAfterInitial).toBeGreaterThan(0);

    // Verify last_synced_line was written
    const db = (
      manager as unknown as {
        db: { prepare: (s: string) => { get: (...a: unknown[]) => unknown } };
      }
    ).db;
    const record = db
      .prepare(`SELECT last_synced_line FROM files WHERE path = ? AND source = ?`)
      .get("sessions/live-convo.jsonl", "sessions") as { last_synced_line: number } | undefined;
    expect(record?.last_synced_line).toBe(INITIAL);

    // -----------------------------------------------------------------------
    // Phase 2 — append 10 new messages
    // -----------------------------------------------------------------------
    const NEW = 10;
    const appendLines: string[] = [];
    for (let i = INITIAL; i < INITIAL + NEW; i++) {
      const role = i % 2 === 0 ? "user" : "assistant";
      appendLines.push(makeSessionLine(role, `Appended message ${i + 1}: ${"y".repeat(40)}`));
    }
    await fs.appendFile(sessionFile, appendLines.join("\n") + "\n");

    // Simulate watcher-triggered dirty state
    const mgr = manager as unknown as { sessionsDirty: boolean; sessionsDirtyFiles: Set<string> };
    mgr.sessionsDirty = true;
    mgr.sessionsDirtyFiles.add(sessionFile);

    const t2 = performance.now();
    await manager.sync({ reason: "session-delta" });
    const t3 = performance.now();

    const statusAfterIncremental = manager.status();
    const chunksAfterIncremental = statusAfterIncremental.chunks;

    const recordAfter = db
      .prepare(`SELECT last_synced_line FROM files WHERE path = ? AND source = ?`)
      .get("sessions/live-convo.jsonl", "sessions") as { last_synced_line: number } | undefined;

    console.log(
      `[live] Incremental sync: ${chunksAfterIncremental} chunks total (+${chunksAfterIncremental - chunksAfterInitial} new), ${(t3 - t2).toFixed(0)} ms`,
    );
    console.log(
      `[live] last_synced_line: ${record?.last_synced_line} → ${recordAfter?.last_synced_line}`,
    );

    // Incremental sync should have added new chunks without touching old ones
    expect(chunksAfterIncremental).toBeGreaterThan(chunksAfterInitial);
    expect(recordAfter?.last_synced_line).toBe(INITIAL + NEW);
  });

  it("returns zero new chunks when transcript is unchanged", async () => {
    if (!ollamaReachable) {
      return;
    }

    const INITIAL = 20;
    await fs.writeFile(sessionFile, buildTranscript(INITIAL));

    const result = await getMemorySearchManager({
      cfg: createCfg({ workspaceDir, storePath, stateDir }),
      agentId: "main",
    });
    const manager = result.manager!;
    await manager.sync({ reason: "live-initial" });

    const statusBefore = manager.status();
    const chunksBefore = statusBefore.chunks;

    // Second sync — nothing changed, but mark dirty to exercise the path
    const mgr = manager as unknown as { sessionsDirty: boolean; sessionsDirtyFiles: Set<string> };
    mgr.sessionsDirty = true;
    mgr.sessionsDirtyFiles.add(sessionFile);

    await manager.sync({ reason: "session-delta" });

    const statusAfter = manager.status();
    expect(statusAfter.chunks).toBe(chunksBefore);
    console.log(`[live] No-op sync: chunk count stable at ${chunksBefore}`);
  });

  it("full re-index after compaction (file replaced with fewer messages)", async () => {
    if (!ollamaReachable) {
      return;
    }

    const INITIAL = 20;
    await fs.writeFile(sessionFile, buildTranscript(INITIAL));

    const result = await getMemorySearchManager({
      cfg: createCfg({ workspaceDir, storePath, stateDir }),
      agentId: "main",
    });
    const manager = result.manager!;
    await manager.sync({ reason: "live-initial" });

    const db = (
      manager as unknown as {
        db: { prepare: (s: string) => { get: (...a: unknown[]) => unknown } };
      }
    ).db;

    // Simulate compaction: replace the file with a shorter transcript
    await fs.writeFile(sessionFile, buildTranscript(5));

    // Targeted post-compaction sync — should do full re-index
    await manager.sync({ reason: "post-compaction", sessionFiles: [sessionFile] });

    const record = db
      .prepare(`SELECT last_synced_line FROM files WHERE path = ? AND source = ?`)
      .get("sessions/live-convo.jsonl", "sessions") as { last_synced_line: number } | undefined;

    expect(record?.last_synced_line).toBe(5);
    console.log(`[live] Post-compaction: last_synced_line reset to ${record?.last_synced_line}`);
  });
});
