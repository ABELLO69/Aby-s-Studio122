import 'dotenv/config';
import dotenv from 'dotenv';
import express from 'express';
import type { Request, Response } from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, MediaResolution, ThinkingLevel, Type } from '@google/genai';
import type { Content, Part } from '@google/genai';
import {
  ARCHVIZ_CORE_SYSTEM_INSTRUCTION,
  ARCHVIZ_SYSTEM_INSTRUCTION,
  buildArchVizComparisonSystemInstruction,
  buildArchVizSystemInstructionWithRules,
} from './server/archvizPrompt.ts';
import {
  COMPARELEARN_BLOCK_MARKER,
  addSingleRuleToPromptFile,
  clearAllRulesInPromptFile,
  deleteRuleFromPromptFile,
  mergeRulesIntoPromptFile,
  readLiveSystemPrompt,
  updateRuleInPromptFile,
  writeLearnedRulesToPromptFile,
} from './server/promptFileStore.ts';
import {
  isDuplicateRule,
  normalizeRuleText,
  type LearnedRuleSource,
} from './src/utils/knowledgeBase.ts';
import {
  collectGeminiApiKeys,
  isQuotaExhausted429Error,
  isRotatableGeminiError,
  markKeyExhausted,
  orderKeyPoolWithCooldown,
  type GeminiApiKeyEntry,
} from './server/apiKeys.ts';
import { parseAndNormalizeCompareResponse } from './server/compareResponse.ts';
import {
  FIRST_CHUNK_TIMEOUT_MS,
  FirstChunkTimeoutError,
  createAttemptAbortController,
  isFirstChunkTimeoutError,
  racePromiseWithAbort,
} from './server/streamHelpers.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT || 3000);
const isProduction = process.env.NODE_ENV === 'production';
const SERVER_TIMEOUT_MS = 0;

function getLiveSystemInstructionRaw(extraLearnedRules?: string[]): string {
  const livePrompt = readLiveSystemPrompt();
  const allRules: string[] = [...livePrompt.learnedRules];
  if (Array.isArray(extraLearnedRules)) {
    for (const raw of extraLearnedRules) {
      if (typeof raw !== 'string') continue;
      const cleaned = normalizeRuleText(raw);
      if (!cleaned) continue;
      if (!isDuplicateRule(cleaned, allRules)) {
        allRules.push(cleaned);
      }
    }
  }

  return buildArchVizSystemInstructionWithRules(
    ARCHVIZ_SYSTEM_INSTRUCTION,
    allRules
  );
}

function getSystemInstructionForLevel(
  level: ThinkingLevel,
  extraLearnedRules?: string[]
): string {
  const baseWithRules = getLiveSystemInstructionRaw(extraLearnedRules);
  if (level === ThinkingLevel.MINIMAL) {
    return baseWithRules;
  }
  if (level === ThinkingLevel.LOW) {
    return `${baseWithRules}\n\n[FONTOS - GYORS (LOW) MÓD: Gondolkodási összefoglaló nélkül, azonnal a tömör, műszakilag pontos kész választ írd ki magyar nyelven!]`;
  }
  if (level === ThinkingLevel.MEDIUM) {
    return `${baseWithRules}\n\n[FONTOS - KIEGYENSÚLYOZOTT (MEDIUM) MÓD: A belső gondolkodási folyamatodat (thoughts) kizárólag magyar nyelven, lényegre törően és tömören fogalmazd meg!]`;
  }
  return `${baseWithRules}\n\n[FONTOS: A belső gondolkodási folyamatodat (thoughts) kizárólag magyar nyelven írd!]`;
}

function readDotEnvMap(): Record<string, string> {
  const envPath = path.resolve(__dirname, '.env');
  try {
    if (fs.existsSync(envPath)) {
      return dotenv.parse(fs.readFileSync(envPath));
    }
  } catch {
    // Ignore .env inspection failures
  }
  return {};
}

function getConfiguredApiKeyPool(): GeminiApiKeyEntry[] {
  return collectGeminiApiKeys(process.env, readDotEnvMap());
}

const MISSING_API_KEY_HINT = 'Aktív kulcs: Nincs beállítva (Settings > Secrets)';

function getAiClient(apiKey: string): GoogleGenAI {
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
      retryOptions: {
        attempts: 1,
      },
    },
  });
}

interface IncomingMessage {
  role: 'user' | 'model';
  text?: string;
  image?: {
    mimeType: string;
    base64: string;
  };
}

type SupportedModelId = 'gemini-4-hydrogen';

interface AnalyzeRequestBody {
  messages: IncomingMessage[];
  model?: string;
  thinkingLevel?: 'MINIMAL' | 'LOW' | 'MEDIUM' | 'HIGH' | 'XHIGH' | string;
  learnedRules?: string[];
}

interface CompareLearnImagePayload {
  mimeType: string;
  base64: string;
}

interface CompareLearnRequestBody {
  originalImage?: CompareLearnImagePayload;
  generatedImage?: CompareLearnImagePayload;
  originalPrompt?: string;
  userNote?: string;
  includeCorrectedPrompt?: boolean;
  existingRules?: string[];
  model?: string;
  thinkingLevel?: string;
  stream?: boolean;
}

function resolveModelId(_model?: string): SupportedModelId {
  return 'gemini-4-hydrogen';
}

interface ResolvedThinkingConfig {
  thinkingLevel: ThinkingLevel;
  includeThoughts: boolean;
}

function resolveThinkingConfigForModel(
  _model: SupportedModelId,
  level?: string
): ResolvedThinkingConfig {
  const normalized = typeof level === 'string' ? level.trim().toUpperCase() : '';
  let requested: 'MINIMAL' | 'LOW' | 'MEDIUM' | 'HIGH' | 'XHIGH' = 'MEDIUM';

  if (
    normalized === 'MINIMAL' ||
    normalized === 'OFF' ||
    normalized === 'NONE' ||
    normalized === '0' ||
    normalized === 'KIKAPCSOLVA'
  ) {
    requested = 'MINIMAL';
  } else if (normalized === 'LOW' || normalized === 'GYORS') {
    requested = 'LOW';
  } else if (normalized === 'MEDIUM' || normalized === 'KIEGYENSULYOZOTT') {
    requested = 'MEDIUM';
  } else if (normalized === 'HIGH' || normalized === 'MELY') {
    requested = 'HIGH';
  } else if (normalized === 'XHIGH' || normalized === 'EXTRA_HIGH') {
    requested = 'XHIGH';
  }

  if (requested === 'MINIMAL' || requested === 'LOW') {
    // Gemini 4 Hydrogen requires thinkingLevel >= LOW; disable thought streaming in LOW mode for instant answer start
    return { thinkingLevel: ThinkingLevel.LOW, includeThoughts: false };
  }
  if (requested === 'XHIGH') {
    return { thinkingLevel: ThinkingLevel.XHIGH, includeThoughts: true };
  }
  if (requested === 'HIGH') {
    return { thinkingLevel: ThinkingLevel.HIGH, includeThoughts: true };
  }
  return { thinkingLevel: ThinkingLevel.MEDIUM, includeThoughts: true };
}

function buildGeminiContents(messages: IncomingMessage[]): Content[] {
  return messages.map((msg) => {
    const parts: Part[] = [];
    if (msg.image?.base64 && msg.image?.mimeType) {
      parts.push({
        inlineData: {
          mimeType: msg.image.mimeType,
          data: msg.image.base64,
        },
      });
    }
    const trimmedText = msg.text?.trim() ?? '';
    if (trimmedText) {
      parts.push({ text: trimmedText });
    } else if (msg.role === 'user' && msg.image) {
      parts.push({
        text: 'Elemezd a mellékelt építészeti képet a rendszerutasítás minden szabálya szerint, és készítsd el a műszaki kivonatot, a másolható kész promptot és a javaslatot.',
      });
    }
    return {
      role: msg.role,
      parts,
    };
  });
}

function unwrapGoogleErrorPayload(rawMessage: string): {
  code?: number;
  status?: string;
  message?: string;
  detailsSummary?: string;
} {
  let current: unknown = rawMessage;
  let code: number | undefined;
  let status: string | undefined;
  let message: string | undefined;
  let detailsSummary: string | undefined;

  for (let depth = 0; depth < 3; depth++) {
    if (typeof current === 'string') {
      const trimmed = current.trim();
      if (!trimmed.startsWith('{')) {
        if (!message) message = trimmed;
        break;
      }
      try {
        current = JSON.parse(trimmed);
      } catch {
        if (!message) message = trimmed;
        break;
      }
    }

    if (current && typeof current === 'object') {
      const obj = current as Record<string, unknown>;
      const nested =
        obj.error && typeof obj.error === 'object'
          ? (obj.error as Record<string, unknown>)
          : obj;

      if (typeof nested.code === 'number') code = nested.code;
      if (typeof nested.status === 'string') status = nested.status;
      if (Array.isArray(nested.details) && nested.details.length > 0) {
        try {
          detailsSummary = JSON.stringify(nested.details);
        } catch {
          // ignore serialization issues
        }
      }
      if (typeof nested.message === 'string') {
        const innerMsg = nested.message.trim();
        if (innerMsg.startsWith('{')) {
          current = innerMsg;
          continue;
        }
        message = innerMsg;
        break;
      }
    }
    break;
  }

  return { code, status, message, detailsSummary };
}

function extractRawGeminiError(
  error: unknown,
  keyEntry?: { label?: string; hint?: string }
): {
  statusCode: number;
  rawStatus: string;
  rawErrorMessage: string;
  apiKeyHint: string;
  keyLabel?: string;
} {
  const sdkStatus =
    error && typeof error === 'object' && 'status' in error && typeof (error as { status?: unknown }).status === 'number'
      ? (error as { status: number }).status
      : undefined;
  const explicitRawStatus =
    error &&
    typeof error === 'object' &&
    'rawStatus' in error &&
    typeof (error as { rawStatus?: unknown }).rawStatus === 'string'
      ? (error as { rawStatus: string }).rawStatus
      : undefined;
  const explicitApiKeyHint =
    error &&
    typeof error === 'object' &&
    'apiKeyHint' in error &&
    typeof (error as { apiKeyHint?: unknown }).apiKeyHint === 'string'
      ? (error as { apiKeyHint: string }).apiKeyHint
      : undefined;
  const explicitKeyLabel =
    error &&
    typeof error === 'object' &&
    'keyLabel' in error &&
    typeof (error as { keyLabel?: unknown }).keyLabel === 'string'
      ? (error as { keyLabel: string }).keyLabel
      : undefined;
  const rawMessageInput =
    error instanceof Error ? error.message : typeof error === 'string' ? error : JSON.stringify(error);

  const unwrapped = unwrapGoogleErrorPayload(rawMessageInput);
  const isTimeout =
    isFirstChunkTimeoutError(error) ||
    (error instanceof Error && (error.name === 'TimeoutError' || error.name === 'AbortError'));

  const statusCode = unwrapped.code ?? sdkStatus ?? (isTimeout ? 504 : 500);
  const rawStatus =
    explicitRawStatus || (unwrapped.status ? `${statusCode} ${unwrapped.status}` : `${statusCode}`);
  const baseMessage = unwrapped.message || rawMessageInput || 'Ismeretlen szerverhiba.';
  const rawErrorMessage = unwrapped.detailsSummary
    ? `${baseMessage}\n\nRészletek: ${unwrapped.detailsSummary}`
    : baseMessage;

  return {
    statusCode,
    rawStatus,
    rawErrorMessage,
    apiKeyHint: explicitApiKeyHint ?? keyEntry?.hint ?? MISSING_API_KEY_HINT,
    keyLabel: explicitKeyLabel ?? keyEntry?.label,
  };
}

async function executeGenerateContentWithRetry(params: {
  ai: GoogleGenAI;
  model: SupportedModelId;
  contents: Content[];
  thinkingLevel: ThinkingLevel;
  includeThoughts?: boolean;
  learnedRules?: string[];
  maxAttempts?: number;
  abortSignal?: AbortSignal;
}): Promise<{ text: string; thoughts: string; finishReason?: string }> {
  const {
    ai,
    model,
    contents,
    thinkingLevel,
    includeThoughts = true,
    learnedRules,
    maxAttempts = 1,
    abortSignal,
  } = params;
  let lastError: unknown;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    if (abortSignal?.aborted) {
      throw new Error('A kérést a kliens vagy az időkorlát megszakította.');
    }
    try {
      const response = await ai.models.generateContent({
        model,
        contents,
        config: {
          abortSignal,
          systemInstruction: getSystemInstructionForLevel(thinkingLevel, learnedRules),
          mediaResolution: MediaResolution.MEDIA_RESOLUTION_MEDIUM,
          thinkingConfig: {
            thinkingLevel,
            includeThoughts,
          },
        },
      });
      const candidate = response.candidates?.[0];
      const parts = candidate?.content?.parts ?? [];
      let answerAcc = '';
      let thoughtAcc = '';
      for (const part of parts) {
        if (typeof part.text !== 'string' || part.text.length === 0) continue;
        if (part.thought) {
          thoughtAcc += part.text;
        } else {
          answerAcc += part.text;
        }
      }
      const finalText = (answerAcc || response.text || '').trim();
      return {
        text: finalText,
        thoughts: thoughtAcc.trim(),
        finishReason: candidate?.finishReason,
      };
    } catch (err: unknown) {
      lastError = err;
      if (abortSignal?.aborted) {
        throw err;
      }
      const errMsg = err instanceof Error ? err.message : String(err);
      const isTransient =
        errMsg.includes('503') ||
        errMsg.includes('UNAVAILABLE') ||
        errMsg.includes('DEADLINE_EXCEEDED') ||
        errMsg.includes('500') ||
        errMsg.includes('ECONNRESET');

      if (!isTransient || attempt === maxAttempts) {
        throw err;
      }
      const delayMs = 600 * attempt;
      await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
  }

  if (lastError) throw lastError;
  return { text: '', thoughts: '' };
}

// ─── Helper: strip data-URI prefix ───────────────────────────────────────────
function stripDataUriPrefix(dataUrlOrBase64: string): string {
  const idx = dataUrlOrBase64.indexOf(',');
  return idx >= 0 ? dataUrlOrBase64.slice(idx + 1) : dataUrlOrBase64;
}

// ─── Helper: round to nearest 16 ─────────────────────────────────────────────
function roundTo16(n: number): number {
  return Math.round(n / 16) * 16;
}

// ─── Helper: scale to max 1024, 16-multiple ───────────────────────────────────
function scaleToMax1024(w: number, h: number): { w: number; h: number } {
  const max = Math.max(w, h);
  if (max <= 1024) return { w: roundTo16(w), h: roundTo16(h) };
  const ratio = 1024 / max;
  return { w: roundTo16(w * ratio), h: roundTo16(h * ratio) };
}

// ─── Helper: clean JSON markdown fences ──────────────────────────────────────
function cleanJsonFences(raw: string): string {
  return raw
    .trim()
    .replace(/^```(?:json|JSON)?\s*/i, '')
    .replace(/\s*```\s*$/i, '')
    .trim();
}

async function createApp() {
  const app = express();
  app.use(express.json({ limit: '25mb' }));
  app.use(express.urlencoded({ limit: '25mb', extended: true }));

  const normalizeRuleSource = (raw: unknown): LearnedRuleSource => {
    if (raw === 'manual' || raw === 'import') return raw;
    return 'auto';
  };

  app.get('/api/rules', (_req: Request, res: Response) => {
    try {
      const livePrompt = readLiveSystemPrompt();
      res.json({
        rules: livePrompt.ruleItems,
        total: livePrompt.ruleItems.length,
        marker: COMPARELEARN_BLOCK_MARKER,
      });
    } catch (err: unknown) {
      res.status(500).json({
        error:
          err instanceof Error
            ? err.message
            : 'Nem sikerült kiolvasni a System Prompt szabályait.',
      });
    }
  });

  app.post('/api/rules', (req: Request, res: Response) => {
    try {
      const body = (req.body ?? {}) as {
        rules?: unknown;
        text?: unknown;
        mode?: unknown;
        source?: unknown;
      };
      const sourceType = normalizeRuleSource(body.source);

      if (Array.isArray(body.rules)) {
        const incomingStrings = body.rules.filter((r): r is string => typeof r === 'string');
        if (body.mode === 'replace') {
          const data = writeLearnedRulesToPromptFile(incomingStrings);
          res.json({
            rules: data.ruleItems,
            addedTexts: data.learnedRules,
            skippedDuplicates: [],
            total: data.ruleItems.length,
          });
          return;
        }
        const { data, addedTexts, skippedDuplicates } = mergeRulesIntoPromptFile(
          incomingStrings,
          sourceType
        );
        res.json({
          rules: data.ruleItems,
          addedTexts,
          skippedDuplicates,
          total: data.ruleItems.length,
        });
        return;
      }

      if (typeof body.text === 'string') {
        const { data, added, message } = addSingleRuleToPromptFile(body.text, sourceType);
        res.status(added ? 200 : 409).json({
          rules: data.ruleItems,
          added,
          addedTexts: added ? [normalizeRuleText(body.text)] : [],
          message,
          total: data.ruleItems.length,
        });
        return;
      }

      res.status(400).json({ error: 'Hiányzó szabálylista ("rules") vagy szöveg ("text").' });
    } catch (err: unknown) {
      res.status(500).json({
        error:
          err instanceof Error
            ? err.message
            : 'Nem sikerült módosítani a System Prompt fájlt.',
      });
    }
  });

  const handleUpdateRuleRoute = (req: Request<{ id: string }>, res: Response) => {
    try {
      const body = (req.body ?? {}) as { text?: unknown; previousText?: unknown };
      if (typeof body.text !== 'string' || !body.text.trim()) {
        res.status(400).json({ error: 'A módosított szabály szövege nem lehet üres.' });
        return;
      }
      const previousText = typeof body.previousText === 'string' ? body.previousText : undefined;
      const { data, updated, message } = updateRuleInPromptFile(
        req.params.id,
        body.text,
        previousText
      );
      res.status(updated ? 200 : 400).json({
        rules: data.ruleItems,
        updated,
        message,
        total: data.ruleItems.length,
      });
    } catch (err: unknown) {
      res.status(500).json({
        error:
          err instanceof Error
            ? err.message
            : 'Nem sikerült frissíteni a szabályt a System Prompt fájlban.',
      });
    }
  };

  app.put('/api/rules/:id', handleUpdateRuleRoute);
  app.patch('/api/rules/:id', handleUpdateRuleRoute);

  app.delete('/api/rules/:id', (req: Request<{ id: string }>, res: Response) => {
    try {
      const idOrText = req.params.id;
      if (idOrText === 'all') {
        const { data } = clearAllRulesInPromptFile();
        res.json({ rules: data.ruleItems, deleted: true, total: 0 });
        return;
      }
      const fallbackText =
        req.body && typeof (req.body as { text?: unknown }).text === 'string'
          ? (req.body as { text: string }).text
          : undefined;
      let result = deleteRuleFromPromptFile(idOrText);
      if (!result.deleted && fallbackText) {
        result = deleteRuleFromPromptFile(fallbackText);
      }
      res.json({
        rules: result.data.ruleItems,
        deleted: result.deleted,
        total: result.data.ruleItems.length,
      });
    } catch (err: unknown) {
      res.status(500).json({
        error:
          err instanceof Error
            ? err.message
            : 'Nem sikerült törölni a szabályt a System Prompt fájlból.',
      });
    }
  });

  app.delete('/api/rules', (_req: Request, res: Response) => {
    try {
      const { data } = clearAllRulesInPromptFile();
      res.json({ rules: data.ruleItems, deleted: true, total: 0 });
    } catch (err: unknown) {
      res.status(500).json({
        error:
          err instanceof Error
            ? err.message
            : 'Nem sikerült kiüríteni a szabályokat a System Prompt fájlból.',
      });
    }
  });

  app.post('/api/analyze', async (req: Request<unknown, unknown, AnalyzeRequestBody>, res: Response) => {
    req.setTimeout(SERVER_TIMEOUT_MS);
    res.setTimeout(SERVER_TIMEOUT_MS);
    const resolvedModel = resolveModelId(req.body?.model);
    let clientAbortController: AbortController | null = null;
    let resolvedThinkingConfig: ResolvedThinkingConfig | undefined;
    let lastKeyEntry: { label?: string; hint?: string } | undefined;
    try {
      const { messages, thinkingLevel, learnedRules } = req.body ?? {};
      if (!Array.isArray(messages) || messages.length === 0) {
        res.status(400).json({ error: 'Hiányzó üzenetlista az elemzéshez.' });
        return;
      }

      const keyPool = getConfiguredApiKeyPool();
      if (keyPool.length === 0) {
        res.status(500).json({
          error: 'A GEMINI_API_KEY nincs konfigurálva a szerveren.',
          rawStatus: '500 MISSING_API_KEY',
          apiKeyHint: MISSING_API_KEY_HINT,
          model: resolvedModel,
        });
        return;
      }

      resolvedThinkingConfig = resolveThinkingConfigForModel(resolvedModel, thinkingLevel);
      const contents = buildGeminiContents(messages);

      clientAbortController = new AbortController();
      res.on('close', () => {
        if (!res.writableEnded) {
          clientAbortController?.abort();
        }
      });

      const { orderedKeys, skippedCooldownKeys } = orderKeyPoolWithCooldown(keyPool);
      if (skippedCooldownKeys.length > 0) {
        console.info(
          `[Gemini Cooldown] Analyze kihagyott 429 kulcs(ok): ${skippedCooldownKeys.map((k) => k.label).join(', ')} -> indítás: ${orderedKeys[0].label}`
        );
      }

      let result: { text: string; thoughts: string; finishReason?: string } | undefined;
      for (let keyIdx = 0; keyIdx < orderedKeys.length; keyIdx++) {
        const keyEntry = orderedKeys[keyIdx];
        lastKeyEntry = keyEntry;
        console.info(
          `[Gemini Analyze] model=${resolvedModel} | requestedLevel=${thinkingLevel ?? 'default'} | learnedRules=${Array.isArray(learnedRules) ? learnedRules.length : 0} | thinkingConfig=${JSON.stringify(resolvedThinkingConfig)} | ${keyEntry.label}`
        );

        try {
          const ai = getAiClient(keyEntry.key);
          result = await executeGenerateContentWithRetry({
            ai,
            model: resolvedModel,
            contents,
            thinkingLevel: resolvedThinkingConfig.thinkingLevel,
            includeThoughts: resolvedThinkingConfig.includeThoughts,
            learnedRules,
            maxAttempts: 1,
            abortSignal: clientAbortController.signal,
          });
          break;
        } catch (attemptError: unknown) {
          if (clientAbortController.signal.aborted) {
            throw attemptError;
          }
          const errInfo = extractRawGeminiError(attemptError, keyEntry);
          if (isQuotaExhausted429Error(errInfo.statusCode, errInfo.rawStatus, errInfo.rawErrorMessage)) {
            markKeyExhausted(keyEntry.key, errInfo.rawStatus);
            console.warn(`[Gemini Cooldown] ${keyEntry.label} 15 percre pihenőre téve (429).`);
          }
          const canRotate =
            isRotatableGeminiError(errInfo.statusCode, errInfo.rawStatus, errInfo.rawErrorMessage) &&
            keyIdx < orderedKeys.length - 1;

          if (canRotate) {
            console.warn(
              `[Gemini Key Rotation] Analyze ${keyEntry.label} hibát kapott (${errInfo.rawStatus}), átváltás: ${orderedKeys[keyIdx + 1].label}`
            );
            continue;
          }
          throw attemptError;
        }
      }

      if (clientAbortController.signal.aborted) {
        if (!res.writableEnded) res.end();
        return;
      }

      const activeKey = lastKeyEntry ?? orderedKeys[0];
      if (!result?.text) {
        res.status(502).json({
          error: result?.finishReason
            ? `A modell üres választ adott vissza (finishReason: ${result.finishReason}).`
            : 'A modell üres választ adott vissza.',
          rawStatus: '502 EMPTY_RESPONSE',
          apiKeyHint: activeKey.hint,
          keyLabel: activeKey.label,
          model: resolvedModel,
          thinkingLevel: resolvedThinkingConfig.thinkingLevel,
          thinkingConfig: resolvedThinkingConfig,
        });
        return;
      }

      res.json({
        text: result.text,
        thoughts: result.thoughts || undefined,
        model: resolvedModel,
        thinkingLevel: resolvedThinkingConfig.thinkingLevel,
        thinkingConfig: resolvedThinkingConfig,
        keyLabel: activeKey.label,
        apiKeyHint: activeKey.hint,
      });
    } catch (error: unknown) {
      if (clientAbortController?.signal.aborted) {
        if (!res.writableEnded) res.end();
        return;
      }
      const { statusCode, rawStatus, rawErrorMessage, apiKeyHint, keyLabel } =
        extractRawGeminiError(error, lastKeyEntry);
      console.error(`Gemini API hiba (${rawStatus}, ${resolvedModel}, ${keyLabel ?? 'nincs kulcs'}): ${rawErrorMessage}`);
      res.status(statusCode).json({
        error: rawErrorMessage,
        rawStatus,
        apiKeyHint,
        keyLabel,
        model: resolvedModel,
        thinkingLevel: resolvedThinkingConfig?.thinkingLevel,
        thinkingConfig: resolvedThinkingConfig,
      });
    }
  });

  app.post('/api/analyze-stream', async (req: Request<unknown, unknown, AnalyzeRequestBody>, res: Response) => {
    req.setTimeout(SERVER_TIMEOUT_MS);
    res.setTimeout(SERVER_TIMEOUT_MS);
    const resolvedModel = resolveModelId(req.body?.model);
    let clientAbortController: AbortController | null = null;
    let pingTimer: ReturnType<typeof setInterval> | undefined;
    let resolvedThinkingConfig: ResolvedThinkingConfig | undefined;
    let lastKeyEntry: { label?: string; hint?: string } | undefined;
    try {
      const { messages, thinkingLevel, learnedRules } = req.body ?? {};
      if (!Array.isArray(messages) || messages.length === 0) {
        res.status(400).json({ error: 'Hiányzó üzenetlista az elemzéshez.' });
        return;
      }

      const keyPool = getConfiguredApiKeyPool();
      if (keyPool.length === 0) {
        res.status(500).json({
          error: 'A GEMINI_API_KEY nincs konfigurálva a szerveren.',
          rawStatus: '500 MISSING_API_KEY',
          apiKeyHint: MISSING_API_KEY_HINT,
          model: resolvedModel,
        });
        return;
      }

      resolvedThinkingConfig = resolveThinkingConfigForModel(resolvedModel, thinkingLevel);
      const contents = buildGeminiContents(messages);

      clientAbortController = new AbortController();
      res.on('close', () => {
        if (!res.writableEnded) {
          clientAbortController?.abort();
        }
      });

      // Start SSE headers + 2048-byte proxy flush + 3s `: ping` keepalive immediately
      res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
      res.setHeader('Cache-Control', 'no-cache, no-transform');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');
      res.flushHeaders?.();

      res.write(':' + ' '.repeat(2048) + '\n\n');

      pingTimer = setInterval(() => {
        if (!res.writableEnded && !clientAbortController?.signal.aborted) {
          res.write(': ping\n\n');
        }
      }, 3000);

      let emittedAnswer = false;
      let lastFinishReason: string | undefined;

      res.write(
        `data: ${JSON.stringify({
          type: 'status',
          step: 1,
          state: 'completed',
          message: 'Kép előkészítve és továbbítva a szervernek',
        })}\n\n`
      );

      const { orderedKeys, skippedCooldownKeys, allInCooldownFallback } =
        orderKeyPoolWithCooldown(keyPool);

      if (skippedCooldownKeys.length > 0) {
        console.info(
          `[Gemini Cooldown] Stream kihagyott 429 kulcs(ok): ${skippedCooldownKeys.map((k) => k.label).join(', ')} -> indítás: ${orderedKeys[0].label}`
        );
      }

      for (let keyIdx = 0; keyIdx < orderedKeys.length; keyIdx++) {
        const keyEntry = orderedKeys[keyIdx];
        lastKeyEntry = keyEntry;
        console.info(
          `[Gemini Stream] model=${resolvedModel} | requestedLevel=${thinkingLevel ?? 'default'} | learnedRules=${Array.isArray(learnedRules) ? learnedRules.length : 0} | thinkingConfig=${JSON.stringify(resolvedThinkingConfig)} | ${keyEntry.label}`
        );

        if (keyIdx === 0) {
          const step2Message =
            skippedCooldownKeys.length > 0
              ? `${skippedCooldownKeys.map((k) => `#${k.index}`).join(', ')}. kulcs kvótapihenőn (429) · indítás -> ${keyEntry.label}`
              : allInCooldownFallback
                ? `Minden kulcs kvótapihenőn volt · újrapróbálás -> ${keyEntry.label}`
                : `Csatlakozás: ${keyEntry.label}`;

          res.write(
            `data: ${JSON.stringify({
              type: 'status',
              step: 2,
              state: 'active',
              message: step2Message,
              keyLabel: keyEntry.label,
              apiKeyHint: keyEntry.hint,
              skippedCooldownKeys: skippedCooldownKeys.length,
            })}\n\n`
          );
        }

        res.write(
          `data: ${JSON.stringify({
            type: 'meta',
            model: resolvedModel,
            thinkingLevel: resolvedThinkingConfig.thinkingLevel,
            thinkingConfig: resolvedThinkingConfig,
            keyLabel: keyEntry.label,
            apiKeyHint: keyEntry.hint,
          })}\n\n`
        );

        let emittedAnyPartForAttempt = false;
        let emittedFirstAnswerForAttempt = false;
        let receivedFirstChunk = false;
        const attemptAbort = createAttemptAbortController(
          clientAbortController.signal,
          FIRST_CHUNK_TIMEOUT_MS
        );
        let streamIterator: AsyncIterator<unknown> | undefined;
        try {
          const ai = getAiClient(keyEntry.key);
          const streamIterable = await racePromiseWithAbort(
            ai.models.generateContentStream({
              model: resolvedModel,
              contents,
              config: {
                abortSignal: attemptAbort.signal,
                systemInstruction: getSystemInstructionForLevel(
                  resolvedThinkingConfig.thinkingLevel,
                  learnedRules
                ),
                mediaResolution: MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                thinkingConfig: {
                  thinkingLevel: resolvedThinkingConfig.thinkingLevel,
                  includeThoughts: resolvedThinkingConfig.includeThoughts,
                },
              },
            }),
            attemptAbort.signal,
            attemptAbort.didTimeoutFirstChunk,
            FIRST_CHUNK_TIMEOUT_MS
          );

          streamIterator = streamIterable[Symbol.asyncIterator]();

          while (true) {
            if (clientAbortController.signal.aborted) break;
            const nextPromise = streamIterator.next();
            const nextStep = receivedFirstChunk
              ? await nextPromise
              : await racePromiseWithAbort(
                  nextPromise,
                  attemptAbort.signal,
                  attemptAbort.didTimeoutFirstChunk,
                  FIRST_CHUNK_TIMEOUT_MS
                );
            if (nextStep.done) break;
            const chunk = nextStep.value as {
              candidates?: Array<{
                finishReason?: string;
                content?: { parts?: Array<{ text?: string; thought?: boolean }> };
              }>;
              text?: string;
            };

            if (clientAbortController.signal.aborted) break;

            if (!receivedFirstChunk) {
              receivedFirstChunk = true;
              attemptAbort.markFirstChunkReceived();
              res.write(
                `data: ${JSON.stringify({
                  type: 'status',
                  step: 2,
                  state: 'completed',
                  message: `Csatlakozva: ${keyEntry.label}`,
                  keyLabel: keyEntry.label,
                  apiKeyHint: keyEntry.hint,
                })}\n\n`
              );
              res.write(
                `data: ${JSON.stringify({
                  type: 'status',
                  step: 3,
                  state: 'active',
                  message: resolvedThinkingConfig.includeThoughts
                    ? 'Gondolkodási folyamat indítása...'
                    : 'Műszaki térelemzés feldolgozása...',
                })}\n\n`
              );
            }

            const candidateFinish = chunk.candidates?.[0]?.finishReason;
            if (candidateFinish) {
              lastFinishReason = candidateFinish;
            }

            let chunkEmittedPart = false;
            const parts = chunk.candidates?.[0]?.content?.parts;
            if (Array.isArray(parts) && parts.length > 0) {
              for (const part of parts) {
                if (typeof part.text !== 'string' || part.text.length === 0) continue;
                chunkEmittedPart = true;
                emittedAnyPartForAttempt = true;
                if (part.thought === true) {
                  res.write(`data: ${JSON.stringify({ type: 'thought', text: part.text })}\n\n`);
                } else {
                  if (!emittedFirstAnswerForAttempt) {
                    emittedFirstAnswerForAttempt = true;
                    res.write(
                      `data: ${JSON.stringify({
                        type: 'status',
                        step: 3,
                        state: 'completed',
                        message: 'Műszaki térelemzés elkészült',
                      })}\n\n`
                    );
                    res.write(
                      `data: ${JSON.stringify({
                        type: 'status',
                        step: 4,
                        state: 'active',
                        message: 'Elemzés és prompt írása...',
                      })}\n\n`
                    );
                  }
                  emittedAnswer = true;
                  res.write(`data: ${JSON.stringify({ type: 'text', text: part.text })}\n\n`);
                }
              }
            }

            if (!chunkEmittedPart && typeof chunk.text === 'string' && chunk.text.length > 0) {
              emittedAnyPartForAttempt = true;
              if (!emittedFirstAnswerForAttempt) {
                emittedFirstAnswerForAttempt = true;
                res.write(
                  `data: ${JSON.stringify({
                    type: 'status',
                    step: 3,
                    state: 'completed',
                    message: 'Műszaki térelemzés elkészült',
                  })}\n\n`
                );
                res.write(
                  `data: ${JSON.stringify({
                    type: 'status',
                    step: 4,
                    state: 'active',
                    message: 'Elemzés és prompt írása...',
                  })}\n\n`
                );
              }
              emittedAnswer = true;
              res.write(`data: ${JSON.stringify({ type: 'text', text: chunk.text })}\n\n`);
            }
          }
          break;
        } catch (attemptError: unknown) {
          if (clientAbortController.signal.aborted) {
            throw attemptError;
          }
          const isTimedOut =
            attemptAbort.didTimeoutFirstChunk() || isFirstChunkTimeoutError(attemptError);
          const normalizedError = isTimedOut
            ? new FirstChunkTimeoutError(FIRST_CHUNK_TIMEOUT_MS)
            : attemptError;
          const errInfo = extractRawGeminiError(normalizedError, keyEntry);
          if (isQuotaExhausted429Error(errInfo.statusCode, errInfo.rawStatus, errInfo.rawErrorMessage)) {
            markKeyExhausted(keyEntry.key, errInfo.rawStatus);
            console.warn(`[Gemini Cooldown] ${keyEntry.label} 15 percre pihenőre téve (429).`);
          }
          const canRotate =
            !emittedAnyPartForAttempt &&
            (isTimedOut ||
              isRotatableGeminiError(errInfo.statusCode, errInfo.rawStatus, errInfo.rawErrorMessage)) &&
            keyIdx < orderedKeys.length - 1;

          if (canRotate) {
            const nextKey = orderedKeys[keyIdx + 1];
            const rotationMsg = isTimedOut
              ? `Szerver lassú / terhelt – azonnali váltás következő kulcsra (${nextKey.label})...`
              : `${keyEntry.label} foglalt (${errInfo.rawStatus}), azonnali váltás -> ${nextKey.label}...`;
            console.warn(
              `[Gemini Key Rotation] Stream ${keyEntry.label} ${isTimedOut ? `${Math.round(FIRST_CHUNK_TIMEOUT_MS / 1000)}s timeout` : errInfo.rawStatus}, azonnali átváltás: ${nextKey.label}`
            );
            res.write(
              `data: ${JSON.stringify({
                type: 'status',
                step: 2,
                state: 'active',
                message: rotationMsg,
                keyLabel: nextKey.label,
                apiKeyHint: nextKey.hint,
                rotatedFromKey: keyEntry.label,
              })}\n\n`
            );
            continue;
          }
          throw normalizedError;
        } finally {
          attemptAbort.cleanup();
          if (!receivedFirstChunk || attemptAbort.signal.aborted) {
            void streamIterator?.return?.().catch(() => {});
          }
        }
      }

      if (clientAbortController.signal.aborted) {
        res.end();
        return;
      }

      const finalKeyEntry = lastKeyEntry ?? orderedKeys[0];
      if (!emittedAnswer) {
        const emptyMsg = lastFinishReason
          ? `A modell üres választ adott vissza (finishReason: ${lastFinishReason}).`
          : 'A modell üres választ adott vissza.';
        res.write(
          `data: ${JSON.stringify({
            type: 'error',
            error: emptyMsg,
            statusCode: 502,
            rawStatus: '502 EMPTY_RESPONSE',
            apiKeyHint: finalKeyEntry.hint,
            keyLabel: finalKeyEntry.label,
            model: resolvedModel,
            thinkingLevel: resolvedThinkingConfig.thinkingLevel,
            thinkingConfig: resolvedThinkingConfig,
          })}\n\n`
        );
      } else {
        res.write(
          `data: ${JSON.stringify({
            type: 'status',
            step: 4,
            state: 'completed',
            message: 'Elemzés és prompt elkészült',
            keyLabel: finalKeyEntry.label,
          })}\n\n`
        );
        res.write(
          `data: ${JSON.stringify({
            type: 'done',
            done: true,
            model: resolvedModel,
            thinkingLevel: resolvedThinkingConfig.thinkingLevel,
            thinkingConfig: resolvedThinkingConfig,
            keyLabel: finalKeyEntry.label,
            apiKeyHint: finalKeyEntry.hint,
          })}\n\n`
        );
      }
      res.end();
    } catch (error: unknown) {
      if (clientAbortController?.signal.aborted) {
        if (!res.writableEnded) {
          res.end();
        }
        return;
      }

      const { statusCode, rawStatus, rawErrorMessage, apiKeyHint, keyLabel } =
        extractRawGeminiError(error, lastKeyEntry);
      console.error(`Gemini API stream hiba (${rawStatus}, ${resolvedModel}, ${keyLabel ?? 'nincs kulcs'}): ${rawErrorMessage}`);

      if (res.headersSent) {
        if (!res.writableEnded) {
          res.write(
            `data: ${JSON.stringify({
              type: 'error',
              error: rawErrorMessage,
              statusCode,
              rawStatus,
              apiKeyHint,
              keyLabel,
              model: resolvedModel,
              thinkingLevel: resolvedThinkingConfig?.thinkingLevel,
              thinkingConfig: resolvedThinkingConfig,
            })}\n\n`
          );
          res.end();
        }
      } else {
        res.status(statusCode).json({
          error: rawErrorMessage,
          rawStatus,
          apiKeyHint,
          keyLabel,
          model: resolvedModel,
          thinkingLevel: resolvedThinkingConfig?.thinkingLevel,
          thinkingConfig: resolvedThinkingConfig,
        });
      }
    } finally {
      if (pingTimer) {
        clearInterval(pingTimer);
      }
    }
  });

  // ══════════════════════════════════════════════════════════════════════════
  // /api/segment-object  — Gemini Magic Click: single-object bounding box
  // ══════════════════════════════════════════════════════════════════════════
  app.post('/api/segment-object', async (req: Request, res: Response) => {
    try {
      const body = req.body as {
        image?: { base64?: string; mimeType?: string };
        clickX?: number;
        clickY?: number;
        apiKey?: string;
      };

      if (!body.image?.base64 || !body.image?.mimeType) {
        res.status(400).json({ error: 'Hiányzik a kép a kérésből.' });
        return;
      }

      const clientApiKey = typeof body.apiKey === 'string' && body.apiKey.trim() ? body.apiKey.trim() : null;
      const keyPool = clientApiKey
        ? [{ key: clientApiKey, label: 'client-key', hint: 'Kliens kulcs' }]
        : getConfiguredApiKeyPool();

      if (keyPool.length === 0) {
        res.status(500).json({ error: 'GEMINI_API_KEY nincs konfigurálva.' });
        return;
      }

      const clickX = typeof body.clickX === 'number' ? body.clickX : 500;
      const clickY = typeof body.clickY === 'number' ? body.clickY : 500;

      const systemPrompt = `You are a precise object segmentation model. The user clicked at approximately x=${clickX}, y=${clickY} on a 0-1000 coordinate scale. Identify the single distinct object or surface at that location. Return ONLY valid JSON: {"label": "<object name>", "box_2d": [ymin, xmin, ymax, xmax]} where values are integers 0-1000. No other text.`;

      const contents: Content[] = [
        {
          role: 'user',
          parts: [
            {
              inlineData: {
                mimeType: body.image.mimeType,
                data: stripDataUriPrefix(body.image.base64),
              },
            },
            {
              text: `Identify the object at click position x=${clickX}, y=${clickY} (0-1000 scale). Return ONLY JSON: {"label": "...", "box_2d": [ymin, xmin, ymax, xmax]}`,
            },
          ],
        },
      ];

      let lastError: unknown;
      for (const keyEntry of keyPool) {
        try {
          const ai = getAiClient(keyEntry.key);
          const response = await ai.models.generateContent({
            model: 'gemini-3.5-flash-lite',
            contents,
            config: {
              systemInstruction: systemPrompt,
              thinkingConfig: { thinkingLevel: ThinkingLevel.MINIMAL, includeThoughts: false },
              responseMimeType: 'application/json',
            },
          });

          let rawText = response.text ?? '';
          rawText = cleanJsonFences(rawText);

          let parsed: unknown;
          try {
            parsed = JSON.parse(rawText);
          } catch {
            const firstBrace = rawText.indexOf('{');
            const lastBrace = rawText.lastIndexOf('}');
            if (firstBrace !== -1 && lastBrace > firstBrace) {
              parsed = JSON.parse(rawText.slice(firstBrace, lastBrace + 1));
            } else {
              throw new Error('Nem érvényes JSON válasz a Gemini-től');
            }
          }

          const obj = parsed as { label?: string; box_2d?: unknown };
          if (!Array.isArray(obj.box_2d) || obj.box_2d.length !== 4) {
            throw new Error('Hiányzó box_2d a válaszból');
          }

          const box2d = (obj.box_2d as number[]).map((v) => Math.min(1000, Math.max(0, Math.round(Number(v)))));
          res.json({ label: obj.label ?? 'object', box_2d: box2d });
          return;
        } catch (err) {
          lastError = err;
          const errInfo = extractRawGeminiError(err, keyEntry);
          if (isQuotaExhausted429Error(errInfo.statusCode, errInfo.rawStatus, errInfo.rawErrorMessage)) {
            markKeyExhausted(keyEntry.key, errInfo.rawStatus);
            res.status(429).json({ error: 'Kvóta kimerült, váltás ecset módra.', rawStatus: errInfo.rawStatus });
            return;
          }
        }
      }

      const finalErr = extractRawGeminiError(lastError);
      res.status(finalErr.statusCode).json({ error: finalErr.rawErrorMessage });
    } catch (err: unknown) {
      const errInfo = extractRawGeminiError(err);
      res.status(errInfo.statusCode).json({ error: errInfo.rawErrorMessage });
    }
  });

  // ══════════════════════════════════════════════════════════════════════════
  // /api/prepare-edit-prompt — Gemini HIGH: craft English PBR edit instruction
  // ══════════════════════════════════════════════════════════════════════════
  app.post('/api/prepare-edit-prompt', async (req: Request, res: Response) => {
    try {
      const body = req.body as {
        image?: { base64?: string; mimeType?: string };
        maskPreview?: { base64?: string; mimeType?: string };
        userInstruction?: string;
        mode?: string;
        apiKey?: string;
      };

      if (!body.image?.base64 || !body.image?.mimeType) {
        res.status(400).json({ error: 'Hiányzik a kép.' });
        return;
      }

      const clientApiKey = typeof body.apiKey === 'string' && body.apiKey.trim() ? body.apiKey.trim() : null;
      const keyPool = clientApiKey
        ? [{ key: clientApiKey, label: 'client-key', hint: 'Kliens kulcs' }]
        : getConfiguredApiKeyPool();

      if (keyPool.length === 0) {
        res.status(500).json({ error: 'GEMINI_API_KEY nincs konfigurálva.' });
        return;
      }

      const mode = body.mode ?? 'remove';
      const userInstruction = body.userInstruction?.trim() ?? '';

      const systemPrompt = `You are a photorealistic architectural visualization specialist and PBR prompt engineer. Your task: given an interior or exterior render image, and optionally a mask preview showing the cyan-highlighted selection area, write a precise English instruction for a reference-based image editing model (FLUX.2 Klein) that will modify ONLY the selected/highlighted region.

RULES:
1. Describe the selected object's exact visual location and context (e.g. "the dark wooden barstool on the right side in front of the white kitchen island").
2. For REMOVAL: Do NOT say "remove X". Instead describe the empty background surface that should appear in its place (e.g. "continuous oak hardwood floor planks matching the surrounding floor, empty wall with skirting board, no furniture").
3. Preserve camera angle, perspective, lighting, shadows, and all surrounding materials with exact pixel alignment and zero camera shift.
4. Keep the instruction under 200 words, in English, technically precise.
5. Return ONLY JSON: {"prompt": "<the instruction>"}`;

      const imageParts: Part[] = [
        {
          inlineData: {
            mimeType: body.image.mimeType,
            data: stripDataUriPrefix(body.image.base64),
          },
        },
      ];

      if (body.maskPreview?.base64 && body.maskPreview?.mimeType) {
        imageParts.push({
          inlineData: {
            mimeType: body.maskPreview.mimeType,
            data: stripDataUriPrefix(body.maskPreview.base64),
          },
        });
      }

      const contents: Content[] = [
        {
          role: 'user',
          parts: [
            ...imageParts,
            {
              text: `Mode: ${mode}. User instruction: ${userInstruction || '(none provided — use your best judgment based on the selected area)'}. Write the PBR editing instruction as JSON: {"prompt": "..."}`,
            },
          ],
        },
      ];

      let lastError: unknown;
      for (const keyEntry of keyPool) {
        try {
          const ai = getAiClient(keyEntry.key);
          const response = await ai.models.generateContent({
            model: 'gemini-3.5-flash-lite',
            contents,
            config: {
              systemInstruction: systemPrompt,
              thinkingConfig: { thinkingLevel: ThinkingLevel.MINIMAL, includeThoughts: false },
            },
          });

          let rawText = response.text ?? '';
          rawText = cleanJsonFences(rawText);
          const firstBrace = rawText.indexOf('{');
          const lastBrace = rawText.lastIndexOf('}');
          if (firstBrace === -1 || lastBrace <= firstBrace) throw new Error('Nem érvényes JSON');
          const parsed = JSON.parse(rawText.slice(firstBrace, lastBrace + 1)) as { prompt?: string };
          res.json({ prompt: parsed.prompt ?? userInstruction });
          return;
        } catch (err) {
          lastError = err;
          const errInfo = extractRawGeminiError(err, keyEntry);
          if (isQuotaExhausted429Error(errInfo.statusCode, errInfo.rawStatus, errInfo.rawErrorMessage)) {
            markKeyExhausted(keyEntry.key, errInfo.rawStatus);
          }
        }
      }

      const finalErr = extractRawGeminiError(lastError);
      res.status(finalErr.statusCode).json({ error: finalErr.rawErrorMessage });
    } catch (err: unknown) {
      const errInfo = extractRawGeminiError(err);
      res.status(errInfo.statusCode).json({ error: errInfo.rawErrorMessage });
    }
  });

  // ══════════════════════════════════════════════════════════════════════════
  // /api/edit-image — Cloudflare Workers AI FLUX.2 Klein image editing
  // ══════════════════════════════════════════════════════════════════════════
  app.post('/api/edit-image', async (req: Request, res: Response) => {
    try {
      const body = req.body as {
        image?: { base64?: string; mimeType?: string; width?: number; height?: number };
        mask?: { base64?: string; mimeType?: string; width?: number; height?: number };
        prompt?: string;
        model?: string;
        cfAccountId?: string;
        cfApiToken?: string;
      };

      if (!body.image?.base64) {
        res.status(400).json({ error: 'Hiányzik a kép.' });
        return;
      }
      if (!body.prompt?.trim()) {
        res.status(400).json({ error: 'Hiányzik az instrukció prompt.' });
        return;
      }

      const accountId = body.cfAccountId?.trim() || process.env.CLOUDFLARE_ACCOUNT_ID;
      const apiToken = body.cfApiToken?.trim() || process.env.CLOUDFLARE_API_TOKEN;

      if (!accountId || !apiToken) {
        res.status(500).json({ error: 'Cloudflare hitelesítő adatok nincsenek konfigurálva.' });
        return;
      }

      const requestedModel = body.model === '4b' ? '4b' : '9b';
      const models = requestedModel === '4b'
        ? ['@cf/black-forest-labs/flux-2-klein-4b', '@cf/black-forest-labs/flux-2-klein-9b']
        : ['@cf/black-forest-labs/flux-2-klein-9b'];

      // Prepare image buffer
      const imageBase64 = stripDataUriPrefix(body.image.base64);
      const imageBuffer = Buffer.from(imageBase64, 'base64');
      const imageMime = body.image.mimeType ?? 'image/png';

      // Prepare mask buffer if provided
      let maskBuffer: Buffer | null = null;
      if (body.mask?.base64) {
        maskBuffer = Buffer.from(stripDataUriPrefix(body.mask.base64), 'base64');
      }

      const { w: width, h: height } = (() => {
        const w = typeof body.image.width === 'number' ? body.image.width : 1024;
        const h = typeof body.image.height === 'number' ? body.image.height : 1024;
        return scaleToMax1024(w, h);
      })();

      let lastCfError: string = 'Cloudflare API hívás sikertelen';

      for (const cfModel of models) {
        try {
          const cfUrl = `https://api.cloudflare.com/client/v4/accounts/${accountId}/ai/run/${cfModel}`;

          // Build FormData using global fetch API types
          const formData = new (globalThis as any).FormData();
          formData.append('prompt', body.prompt!.trim());
          formData.append('width', String(width));
          formData.append('height', String(height));

          const imageBlob = new Blob([imageBuffer as any], { type: imageMime });
          formData.append('input_image_0', imageBlob, 'input.png');

          if (maskBuffer) {
            const maskBlob = new Blob([maskBuffer as any], { type: 'image/png' });
            formData.append('mask_image_0', maskBlob, 'mask.png');
          }

          console.info(`[Cloudflare AI] Calling ${cfModel} | prompt: ${body.prompt!.trim().slice(0, 80)}...`);

          const cfResponse = await fetch(cfUrl, {
            method: 'POST',
            headers: {
              Authorization: `Bearer ${apiToken}`,
              // DO NOT set Content-Type — let fetch set it with boundary
            },
            body: formData as unknown as BodyInit,
          });

          const contentType = cfResponse.headers.get('content-type') ?? '';

          if (!cfResponse.ok) {
            const errText = await cfResponse.text().catch(() => '');
            let errMsg = `Cloudflare ${cfResponse.status}: ${errText.slice(0, 300)}`;
            try {
              const j = JSON.parse(errText) as { errors?: { message?: string }[] };
              if (j.errors?.[0]?.message) errMsg = j.errors[0].message;
            } catch { /* ignore */ }

            // If 4b fails with input_image_0 schema error, try 9b
            if (cfModel.includes('4b') && (errMsg.includes('input_image_0') || cfResponse.status === 400)) {
              console.warn(`[Cloudflare AI] 4b schema error, falling back to 9b: ${errMsg}`);
              lastCfError = errMsg;
              continue;
            }
            throw new Error(errMsg);
          }

          // Handle both JSON and binary image responses
          if (contentType.includes('application/json')) {
            const jsonResult = await cfResponse.json() as { result?: { image?: string }; success?: boolean };
            if (jsonResult.result?.image) {
              const b64 = jsonResult.result.image;
              res.json({ image: `data:image/png;base64,${b64}` });
              return;
            }
            throw new Error('Cloudflare válasz nem tartalmaz képet');
          } else {
            // Binary image response
            const arrayBuffer = await cfResponse.arrayBuffer();
            const b64 = Buffer.from(arrayBuffer).toString('base64');
            res.json({ image: `data:image/png;base64,${b64}` });
            return;
          }
        } catch (err) {
          lastCfError = err instanceof Error ? err.message : String(err);
          if (!cfModel.includes('4b')) throw err; // don't retry 9b
        }
      }

      res.status(502).json({ error: lastCfError });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Ismeretlen Cloudflare hiba';
      console.error('[Cloudflare AI] Error:', msg);
      res.status(502).json({ error: msg });
    }
  });

  app.post('/api/compare-learn', async (req: Request<unknown, unknown, CompareLearnRequestBody>, res: Response) => {
    req.setTimeout(SERVER_TIMEOUT_MS);
    res.setTimeout(SERVER_TIMEOUT_MS);
    const requestedModel = resolveModelId(req.body?.model);
    let activeModel: SupportedModelId = requestedModel;
    let clientAbortController: AbortController | null = null;
    let pingTimer: ReturnType<typeof setInterval> | undefined;
    let resolvedThinkingConfig: ResolvedThinkingConfig | undefined;
    let lastKeyEntry: { label?: string; hint?: string } | undefined;
    const wantsSse =
      req.body?.stream === true ||
      (typeof req.headers.accept === 'string' && req.headers.accept.includes('text/event-stream'));

    const emitSseEvent = (payload: Record<string, unknown>) => {
      if (!wantsSse || res.writableEnded || clientAbortController?.signal.aborted) return;
      res.write(`data: ${JSON.stringify(payload)}\n\n`);
    };

    try {
      const {
        originalImage,
        generatedImage,
        originalPrompt,
        userNote,
        includeCorrectedPrompt = false,
        existingRules = [],
        thinkingLevel,
      } = req.body ?? {};

      if (!originalImage?.base64 || !originalImage?.mimeType) {
        res.status(400).json({ error: 'Hiányzik az eredeti tervrajz vagy 3D modell kép.' });
        return;
      }
      if (!generatedImage?.base64 || !generatedImage?.mimeType) {
        res.status(400).json({ error: 'Hiányzik a külső generátorral készült kép.' });
        return;
      }

      const keyPool = getConfiguredApiKeyPool();
      if (keyPool.length === 0) {
        res.status(500).json({
          error: 'A GEMINI_API_KEY nincs konfigurálva a szerveren.',
          rawStatus: '500 MISSING_API_KEY',
          apiKeyHint: MISSING_API_KEY_HINT,
          model: requestedModel,
        });
        return;
      }

      clientAbortController = new AbortController();
      res.on('close', () => {
        if (!res.writableEnded) {
          clientAbortController?.abort();
        }
      });

      if (wantsSse) {
        res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
        res.setHeader('Cache-Control', 'no-cache, no-transform');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('X-Accel-Buffering', 'no');
        res.flushHeaders?.();
        res.write(':' + ' '.repeat(2048) + '\n\n');

        pingTimer = setInterval(() => {
          if (!res.writableEnded && !clientAbortController?.signal.aborted) {
            res.write(': ping\n\n');
          }
        }, 3000);

        emitSseEvent({
          type: 'status',
          step: 1,
          state: 'completed',
          message: 'Eredeti és generált kép előkészítve és továbbítva',
        });
      }

      const livePrompt = readLiveSystemPrompt();
      const mergedCompareRules: string[] = [...livePrompt.learnedRules];
      if (Array.isArray(existingRules)) {
        for (const raw of existingRules) {
          if (typeof raw !== 'string') continue;
          const cleaned = normalizeRuleText(raw);
          if (!cleaned) continue;
          if (!isDuplicateRule(cleaned, mergedCompareRules)) {
            mergedCompareRules.push(cleaned);
            if (mergedCompareRules.length >= 120) break;
          }
        }
      }

      const existingRulesBlock =
        mergedCompareRules.length > 0
          ? mergedCompareRules.map((r, idx) => `${idx + 1}. ${r.trim()}`).join('\n')
          : '(Még nincs korábban elmentett szabály.)';

      const comparisonSystemInstruction = buildArchVizComparisonSystemInstruction(
        mergedCompareRules,
        livePrompt.coreInstruction
      );

      const comparisonInstructionText = [
        'Hasonlítsd össze elfogulatlanul, kíméletlenül szigorúan és tételesen az 1. KÉPET (Eredeti tervrajz / 3D modell) a 2. KÉPPEL (Generált látványterv) a System Prompt és a tanult szabályok minden előírása alapján.',
        '',
        `Javított prompt kérése: ${includeCorrectedPrompt ? 'IGEN' : 'NEM'}`,
        '',
        userNote && userNote.trim()
          ? `Felhasználó megjegyzése a hibához:\n${userNote.trim()}`
          : 'Felhasználó megjegyzése a hibához: (nincs megadva)',
        '',
        'Eredetileg használt prompt:',
        originalPrompt?.trim() || '(nincs megadva)',
        '',
        'Már meglévő belső tudásbázis-szabályok (ezeket ne ismételd feleslegesen, csak valóban új szabályokat fogalmazz meg):',
        existingRulesBlock,
      ].join('\n');

      // Gemini 4 Hydrogen execution for CompareLearn

      const contents: Content[] = [
        {
          role: 'user',
          parts: [
            { text: '1. KÉP — EREDETI TERVRAJZ / 3D MODELL:' },
            {
              inlineData: {
                mimeType: originalImage.mimeType,
                data: originalImage.base64,
              },
            },
            { text: '2. KÉP — GENERÁLT KÉP (ELLENŐRIZENDŐ):' },
            {
              inlineData: {
                mimeType: generatedImage.mimeType,
                data: generatedImage.base64,
              },
            },
            { text: comparisonInstructionText },
          ],
        },
      ];

      const activeKeyPoolResult = orderKeyPoolWithCooldown(keyPool);
      const { orderedKeys, skippedCooldownKeys, allInCooldownFallback } = activeKeyPoolResult;
      resolvedThinkingConfig = resolveThinkingConfigForModel(activeModel, thinkingLevel);

      if (skippedCooldownKeys.length > 0) {
        console.info(
          `[Gemini Cooldown] Compare (${activeModel}) kihagyott 429 kulcs(ok): ${skippedCooldownKeys.map((k) => k.label).join(', ')} -> indítás: ${orderedKeys[0].label}`
        );
      }

      let parsedValidResult:
        | {
            errors: string[];
            learnedRules: string[];
            correctedPrompt: string;
          }
        | undefined;
      let lastAttemptError: unknown;

      for (let keyIdx = 0; keyIdx < orderedKeys.length; keyIdx++) {
        const keyEntry = orderedKeys[keyIdx];
        lastKeyEntry = keyEntry;
        console.info(
          `[Gemini CompareLearn] model=${activeModel} | includeCorrectedPrompt=${Boolean(includeCorrectedPrompt)} | thinkingConfig=${JSON.stringify(resolvedThinkingConfig)} | ${keyEntry.label}`
        );

        if (keyIdx === 0) {
          const step2Message =
            skippedCooldownKeys.length > 0
              ? `${skippedCooldownKeys.map((k) => `#${k.index}`).join(', ')}. kulcs kvótapihenőn (429) · indítás -> ${keyEntry.label} · ${activeModel} · ${resolvedThinkingConfig.thinkingLevel}`
              : allInCooldownFallback
                ? `Minden kulcs kvótapihenőn volt · újrapróbálás -> ${keyEntry.label} · ${activeModel} · ${resolvedThinkingConfig.thinkingLevel}`
                : `${keyEntry.label} · ${activeModel} · thinking: ${resolvedThinkingConfig.thinkingLevel}`;

          emitSseEvent({
            type: 'status',
            step: 2,
            state: 'active',
            message: step2Message,
            keyLabel: keyEntry.label,
            apiKeyHint: keyEntry.hint,
            model: activeModel,
            thinkingLevel: resolvedThinkingConfig.thinkingLevel,
            skippedCooldownKeys: skippedCooldownKeys.length,
          });
        }

        const attemptAbort = createAttemptAbortController(
          clientAbortController.signal,
          FIRST_CHUNK_TIMEOUT_MS
        );
        let receivedFirstChunk = false;
        let streamIterator: AsyncIterator<unknown> | undefined;

        try {
          const ai = getAiClient(keyEntry.key);
          const streamIterable = await racePromiseWithAbort(
            ai.models.generateContentStream({
              model: activeModel,
              contents,
              config: {
                abortSignal: attemptAbort.signal,
                systemInstruction: comparisonSystemInstruction,
                mediaResolution: MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                maxOutputTokens: 65536,
                thinkingConfig: {
                  thinkingLevel: resolvedThinkingConfig.thinkingLevel,
                  includeThoughts: false,
                },
                responseMimeType: 'application/json',
                responseSchema: {
                  type: Type.OBJECT,
                  properties: {
                    errors: {
                      type: Type.ARRAY,
                      items: { type: Type.STRING },
                      description:
                        'Tételes lista minden egyes feltárt hibáról és eltérésről a generált képen az eredetihez és a fő System Prompt szabályaihoz képest.',
                    },
                    learnedRules: {
                      type: Type.ARRAY,
                      items: { type: Type.STRING },
                      description:
                        'Tömör, általános érvényű, pozitív megfogalmazású műszaki szabályok a jövőbeli promptok pontosításához.',
                    },
                    correctedPrompt: {
                      type: Type.STRING,
                      description:
                        'Opcionális mező: kizárólag akkor kitöltve a fő System Prompt 5 blokkjával, ha a javított prompt kérése IGEN; egyébként üres string.',
                    },
                  },
                  required: ['errors', 'learnedRules'],
                },
              },
            }),
            attemptAbort.signal,
            attemptAbort.didTimeoutFirstChunk,
            FIRST_CHUNK_TIMEOUT_MS
          );

          streamIterator = streamIterable[Symbol.asyncIterator]();

          let answerAcc = '';
          let lastFinishReason: string | undefined;
          let emittedStep4Active = false;

          while (true) {
            if (clientAbortController.signal.aborted) break;
            const nextPromise = streamIterator.next();
            const nextStep = receivedFirstChunk
              ? await nextPromise
              : await racePromiseWithAbort(
                  nextPromise,
                  attemptAbort.signal,
                  attemptAbort.didTimeoutFirstChunk,
                  FIRST_CHUNK_TIMEOUT_MS
                );
            if (nextStep.done) break;
            const chunk = nextStep.value as {
              candidates?: Array<{
                finishReason?: string;
                content?: { parts?: Array<{ text?: string; thought?: boolean }> };
              }>;
              text?: string;
            };

            if (clientAbortController.signal.aborted) break;

            if (!receivedFirstChunk) {
              receivedFirstChunk = true;
              attemptAbort.markFirstChunkReceived();
              emitSseEvent({
                type: 'status',
                step: 2,
                state: 'completed',
                message: `Csatlakozva: ${keyEntry.label} · ${activeModel} · thinking: ${resolvedThinkingConfig.thinkingLevel}`,
                keyLabel: keyEntry.label,
                apiKeyHint: keyEntry.hint,
                model: activeModel,
                thinkingLevel: resolvedThinkingConfig.thinkingLevel,
              });
              emitSseEvent({
                type: 'status',
                step: 3,
                state: 'active',
                message: 'Eredeti és generált kép tételes vizuális összevetése...',
                keyLabel: keyEntry.label,
                model: activeModel,
                thinkingLevel: resolvedThinkingConfig.thinkingLevel,
              });
            }

            const finishReason = chunk.candidates?.[0]?.finishReason;
            if (finishReason) {
              lastFinishReason = finishReason;
            }

            let chunkHadPart = false;
            const parts = chunk.candidates?.[0]?.content?.parts;
            if (Array.isArray(parts) && parts.length > 0) {
              for (const part of parts) {
                if (typeof part.text !== 'string' || part.text.length === 0) continue;
                chunkHadPart = true;
                if (!part.thought) {
                  if (!emittedStep4Active) {
                    emittedStep4Active = true;
                    emitSseEvent({
                      type: 'status',
                      step: 3,
                      state: 'completed',
                      message: 'Vizuális összevetés elkészült',
                    });
                    emitSseEvent({
                      type: 'status',
                      step: 4,
                      state: 'active',
                      message: 'Hibák, szabályok és javított prompt összeállítása...',
                    });
                  }
                  answerAcc += part.text;
                }
              }
            }

            if (!chunkHadPart && typeof chunk.text === 'string' && chunk.text.length > 0) {
              if (!emittedStep4Active) {
                emittedStep4Active = true;
                emitSseEvent({
                  type: 'status',
                  step: 3,
                  state: 'completed',
                  message: 'Vizuális összevetés elkészült',
                });
                emitSseEvent({
                  type: 'status',
                  step: 4,
                  state: 'active',
                  message: 'Hibák, szabályok és javított prompt összeállítása...',
                });
              }
              answerAcc += chunk.text;
            }
          }

          if (clientAbortController.signal.aborted) {
            break;
          }

          const rawJsonText = answerAcc.trim();
          if (!rawJsonText) {
            const finishMsg = lastFinishReason
              ? `A modell üres választ adott vissza (finishReason: ${lastFinishReason}).`
              : 'A modell üres választ adott vissza az összehasonlításhoz.';
            throw Object.assign(new Error(finishMsg), {
              status: 502,
              isRetryableModelOutput: true,
            });
          }

          parsedValidResult = parseAndNormalizeCompareResponse(
            rawJsonText,
            Boolean(includeCorrectedPrompt)
          );
          break;
        } catch (attemptError: unknown) {
          if (clientAbortController.signal.aborted) {
            throw attemptError;
          }
          const isTimedOut =
            attemptAbort.didTimeoutFirstChunk() || isFirstChunkTimeoutError(attemptError);
          const normalizedError = isTimedOut
            ? new FirstChunkTimeoutError(FIRST_CHUNK_TIMEOUT_MS)
            : attemptError;
          lastAttemptError = normalizedError;
          const errInfo = extractRawGeminiError(normalizedError, keyEntry);

          if (isQuotaExhausted429Error(errInfo.statusCode, errInfo.rawStatus, errInfo.rawErrorMessage)) {
            markKeyExhausted(keyEntry.key, errInfo.rawStatus);
            console.warn(`[Gemini Cooldown] ${keyEntry.label} 15 percre pihenőre téve (429).`);
          }

          const isInvalidModelJsonOrEmpty = Boolean(
            attemptError &&
              typeof attemptError === 'object' &&
              (('isInvalidJson' in attemptError &&
                (attemptError as { isInvalidJson?: boolean }).isInvalidJson) ||
                ('isRetryableModelOutput' in attemptError &&
                  (attemptError as { isRetryableModelOutput?: boolean }).isRetryableModelOutput))
          );

          const isRetryable =
            isTimedOut ||
            isRotatableGeminiError(
              errInfo.statusCode,
              errInfo.rawStatus,
              errInfo.rawErrorMessage
            ) ||
            isInvalidModelJsonOrEmpty;

          if (isRetryable && keyIdx < orderedKeys.length - 1) {
            const nextKey = orderedKeys[keyIdx + 1];
            const rotationMsg = isTimedOut
              ? `Szerver lassú / terhelt – azonnali váltás következő kulcsra (${nextKey.label})...`
              : `${keyEntry.label} (${errInfo.rawStatus}) — azonnali átváltás: ${nextKey.label} · ${activeModel}`;
            console.warn(
              `[Gemini Key Rotation] Compare (${activeModel}) ${keyEntry.label} ${isTimedOut ? `${Math.round(FIRST_CHUNK_TIMEOUT_MS / 1000)}s timeout` : errInfo.rawStatus}, azonnali átváltás: ${nextKey.label}`
            );
            emitSseEvent({
              type: 'status',
              step: 2,
              state: 'active',
              message: rotationMsg,
              keyLabel: nextKey.label,
              apiKeyHint: nextKey.hint,
              model: activeModel,
              thinkingLevel: resolvedThinkingConfig.thinkingLevel,
              rotatedFromKey: keyEntry.label,
            });
            continue;
          }

          throw normalizedError;
        } finally {
          attemptAbort.cleanup();
          if (!receivedFirstChunk || attemptAbort.signal.aborted) {
            void streamIterator?.return?.().catch(() => {});
          }
        }
      }

      if (clientAbortController.signal.aborted) {
        if (!res.writableEnded) res.end();
        return;
      }

      if (!parsedValidResult) {
        throw lastAttemptError ?? new Error('Az összehasonlító elemzés minden API-kulccsal sikertelen volt.');
      }

      if (Array.isArray(parsedValidResult.learnedRules) && parsedValidResult.learnedRules.length > 0) {
        try {
          const { addedTexts } = mergeRulesIntoPromptFile(
            parsedValidResult.learnedRules,
            'auto'
          );
          if (addedTexts.length > 0) {
            console.info(
              `[Hibaelemző Műhely] ${addedTexts.length} új szabály automatikusan elmentve a learned_rules.json fájlba:`,
              addedTexts
            );
          }
        } catch (saveErr) {
          console.error('[Hibaelemző Műhely] Nem sikerült elmenteni a szabályokat a learned_rules.json fájlba:', saveErr);
        }
      }

      const activeKey = lastKeyEntry ?? keyPool[0];
      const compareThinkingConfig = resolvedThinkingConfig
        ? { thinkingLevel: resolvedThinkingConfig.thinkingLevel, includeThoughts: false }
        : undefined;
      const finalPayload = {
        errors: parsedValidResult.errors,
        learnedRules: parsedValidResult.learnedRules,
        correctedPrompt: parsedValidResult.correctedPrompt || undefined,
        includeCorrectedPrompt: Boolean(includeCorrectedPrompt),
        model: activeModel,
        thinkingLevel: resolvedThinkingConfig?.thinkingLevel,
        thinkingConfig: compareThinkingConfig,
        keyLabel: activeKey.label,
        apiKeyHint: activeKey.hint,
      };

      if (wantsSse) {
        emitSseEvent({
          type: 'status',
          step: 3,
          state: 'completed',
          message: 'Vizuális összevetés elkészült',
        });
        emitSseEvent({
          type: 'status',
          step: 4,
          state: 'completed',
          message: 'Hibák, szabályok és javított prompt összeállítva',
        });
        emitSseEvent({
          type: 'result',
          data: finalPayload,
        });
        res.end();
      } else {
        res.json(finalPayload);
      }
    } catch (error: unknown) {
      if (clientAbortController?.signal.aborted) {
        if (!res.writableEnded) res.end();
        return;
      }
      const { statusCode, rawStatus, rawErrorMessage, apiKeyHint, keyLabel } =
        extractRawGeminiError(error, lastKeyEntry);
      const compareErrorThinkingConfig = resolvedThinkingConfig
        ? { thinkingLevel: resolvedThinkingConfig.thinkingLevel, includeThoughts: false }
        : undefined;
      console.error(`Gemini CompareLearn API hiba (${rawStatus}, ${activeModel}, ${keyLabel ?? 'nincs kulcs'}): ${rawErrorMessage}`);
      if (wantsSse && res.headersSent) {
        if (!res.writableEnded) {
          res.write(
            `data: ${JSON.stringify({
              type: 'error',
              error: rawErrorMessage,
              statusCode,
              rawStatus,
              apiKeyHint,
              keyLabel,
              model: activeModel,
              thinkingLevel: resolvedThinkingConfig?.thinkingLevel,
              thinkingConfig: compareErrorThinkingConfig,
            })}\n\n`
          );
          res.end();
        }
      } else {
        res.status(statusCode).json({
          error: rawErrorMessage,
          rawStatus,
          apiKeyHint,
          keyLabel,
          model: activeModel,
          thinkingLevel: resolvedThinkingConfig?.thinkingLevel,
          thinkingConfig: compareErrorThinkingConfig,
        });
      }
    } finally {
      if (pingTimer) {
        clearInterval(pingTimer);
      }
    }
  });

  if (!isProduction) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  const httpServer = app.listen(PORT, '0.0.0.0', () => {
    const keyPool = getConfiguredApiKeyPool();
    const keySummary =
      keyPool.length > 0
        ? `Konfigurált Gemini kulcsok (${keyPool.length} db): ${keyPool.map((k) => k.label).join(', ')}`
        : MISSING_API_KEY_HINT;
    console.log(
      `ArchViz Prompt Studio szerver elindult: http://0.0.0.0:${PORT} | ${keySummary}`
    );
  });
  httpServer.timeout = SERVER_TIMEOUT_MS;
  httpServer.requestTimeout = SERVER_TIMEOUT_MS;
  httpServer.keepAliveTimeout = 300_000;
  httpServer.headersTimeout = 305_000;
}

createApp();
