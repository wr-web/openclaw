import fs from "node:fs/promises";
import path from "node:path";
import { resolveSessionTranscriptsDirForAgent } from "../config/sessions/paths.js";
import { redactSensitiveText } from "../logging/redact.js";
import { createSubsystemLogger } from "../logging/subsystem.js";
import { hashText } from "./internal.js";

const log = createSubsystemLogger("memory");

export type SessionFileEntry = {
  path: string;
  absPath: string;
  mtimeMs: number;
  size: number;
  hash: string;
  content: string;
  /** Maps each content line (0-indexed) to its 1-indexed JSONL source line. */
  lineMap: number[];
};

export async function listSessionFilesForAgent(agentId: string): Promise<string[]> {
  const dir = resolveSessionTranscriptsDirForAgent(agentId);
  try {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    return entries
      .filter((entry) => entry.isFile())
      .map((entry) => entry.name)
      .filter((name) => name.endsWith(".jsonl"))
      .map((name) => path.join(dir, name));
  } catch {
    return [];
  }
}

export function sessionPathForFile(absPath: string): string {
  return path.join("sessions", path.basename(absPath)).replace(/\\/g, "/");
}

function normalizeSessionText(value: string): string {
  return value
    .replace(/\s*\n+\s*/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function extractSessionText(content: unknown): string | null {
  if (typeof content === "string") {
    const normalized = normalizeSessionText(content);
    return normalized ? normalized : null;
  }
  if (!Array.isArray(content)) {
    return null;
  }
  const parts: string[] = [];
  for (const block of content) {
    if (!block || typeof block !== "object") {
      continue;
    }
    const record = block as { type?: unknown; text?: unknown };
    if (record.type !== "text" || typeof record.text !== "string") {
      continue;
    }
    const normalized = normalizeSessionText(record.text);
    if (normalized) {
      parts.push(normalized);
    }
  }
  if (parts.length === 0) {
    return null;
  }
  return parts.join(" ");
}

/**
 * Parse a session JSONL file and return all user/assistant message lines as
 * `{ collected, lineMap }`. Each entry in `collected` is a `"Role: text"`
 * string; the corresponding `lineMap` entry is the 1-indexed JSONL line number.
 */
function parseSessionLines(raw: string): { collected: string[]; lineMap: number[] } {
  const lines = raw.split("\n");
  const collected: string[] = [];
  const lineMap: number[] = [];
  for (let jsonlIdx = 0; jsonlIdx < lines.length; jsonlIdx++) {
    const line = lines[jsonlIdx];
    if (!line.trim()) {
      continue;
    }
    let record: unknown;
    try {
      record = JSON.parse(line);
    } catch {
      continue;
    }
    if (
      !record ||
      typeof record !== "object" ||
      (record as { type?: unknown }).type !== "message"
    ) {
      continue;
    }
    const message = (record as { message?: unknown }).message as
      | { role?: unknown; content?: unknown }
      | undefined;
    if (!message || typeof message.role !== "string") {
      continue;
    }
    if (message.role !== "user" && message.role !== "assistant") {
      continue;
    }
    const text = extractSessionText(message.content);
    if (!text) {
      continue;
    }
    const safe = redactSensitiveText(text, { mode: "tools" });
    const label = message.role === "user" ? "User" : "Assistant";
    collected.push(`${label}: ${safe}`);
    lineMap.push(jsonlIdx + 1);
  }
  return { collected, lineMap };
}

/**
 * Build a partial session entry containing only content lines starting at
 * `fromContentLine` (0-indexed offset into the full content-line array).
 *
 * Used by incremental session sync so only newly-appended messages are
 * re-embedded rather than the entire transcript. Returns `null` when:
 * - the file is missing or unreadable, or
 * - total message count has not grown past `fromContentLine` (no new lines, or
 *   the file was compacted/truncated — callers should fall back to a full sync).
 */
export async function buildPartialSessionEntry(
  absPath: string,
  fromContentLine: number,
): Promise<SessionFileEntry | null> {
  if (fromContentLine <= 0) {
    return buildSessionEntry(absPath);
  }
  try {
    const stat = await fs.stat(absPath);
    const raw = await fs.readFile(absPath, "utf-8");
    const { collected, lineMap } = parseSessionLines(raw);
    if (collected.length <= fromContentLine) {
      // No new lines beyond the already-synced offset (or content shrank).
      return null;
    }
    const newCollected = collected.slice(fromContentLine);
    const newLineMap = lineMap.slice(fromContentLine);
    const content = newCollected.join("\n");
    return {
      path: sessionPathForFile(absPath),
      absPath,
      mtimeMs: stat.mtimeMs,
      size: stat.size,
      hash: hashText(content + "\n" + newLineMap.join(",")),
      content,
      lineMap: newLineMap,
    };
  } catch (err) {
    log.debug(`Failed reading partial session file ${absPath}: ${String(err)}`);
    return null;
  }
}

export async function buildSessionEntry(absPath: string): Promise<SessionFileEntry | null> {
  try {
    const stat = await fs.stat(absPath);
    const raw = await fs.readFile(absPath, "utf-8");
    const { collected, lineMap } = parseSessionLines(raw);
    const content = collected.join("\n");
    return {
      path: sessionPathForFile(absPath),
      absPath,
      mtimeMs: stat.mtimeMs,
      size: stat.size,
      hash: hashText(content + "\n" + lineMap.join(",")),
      content,
      lineMap,
    };
  } catch (err) {
    log.debug(`Failed reading session file ${absPath}: ${String(err)}`);
    return null;
  }
}
