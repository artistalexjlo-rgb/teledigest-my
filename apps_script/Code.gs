/**
 * MULTYSPEAK ZERO-TOUCH MINER v3.1
 * Apps Script: Drive samples → Gemini → Firestore
 *
 *   Wisdom (мухи)  → wisdom_base   (для МОЗГ-ассистента)
 *   Stories (котлеты) → telegram_queue (очередь автопостинга)
 *
 * SETUP (one-time):
 *   1. File → Project properties → Script properties → Add row:
 *        key:   GEMINI_API_KEY
 *        value: <ваш Gemini API key>
 *      Опционально можно переопределить FIREBASE_PROJECT_ID и FOLDER_ID
 *      теми же script-properties.
 *   2. Запусти `processNewLogs` из редактора вручную — Гугл попросит
 *      авторизовать scopes (Drive, Firestore, External requests).
 *   3. Triggers (значок будильника слева) → Add Trigger:
 *        Function: processNewLogs
 *        Event source: Time-driven
 *        Type: Hour timer / Every hour
 *
 * RE-PROCESSING:
 *   FORCE_REPROCESS=true прогоняет уже помеченные файлы. Idempotency
 *   обеспечивается deterministic doc ID в Firestore — повторные запуски
 *   на тех же patterns не создадут дубликатов (вернётся 409 ALREADY_EXISTS,
 *   что мы трактуем как успех).
 */

// --- Defaults (override via Script Properties if needed) ---
var DEFAULT_FIREBASE_PROJECT_ID = "project-56cb62a9-8914-4ae3-b44";
var DEFAULT_FOLDER_ID = "16cEzGQy0ThTmTm_U3yoLi8uDURqHiFJf";
// Alternated to spread load — preview tiers throw 503 under burst.
// Each runMining_ call picks the next model via _modelCallCounter.
var MODELS = [
  "gemini-3.1-flash-lite-preview",
  "gemini-2.5-flash-lite",
  "gemini-2.5-flash"
];
var _modelCallCounter = 0;
var COLLECTION_AI = "wisdom_base";
var COLLECTION_TG = "telegram_queue";
var FORCE_REPROCESS = false;

// --- Runtime tuning ---
// Apps Script consumer hard limit = 6 min per execution. We exit at 5 min to
// give the in-flight Gemini call time to wrap. Unprocessed files survive
// (not marked) and get picked up by the next 15-min trigger.
var MAX_RUNTIME_MS = 5 * 60 * 1000;
// Free tier Gemini 3.1-flash-lite-preview = 15 RPM. 4.5s gap → ~13 RPM,
// safely under cap, leaves headroom for retries.
var INTER_FILE_PAUSE_MS = 4500;
// Retry schedule for transient Gemini failures (503/429/UNAVAILABLE).
// 5s → 20s → 60s. Total worst case ~85s of waits per file.
var GEMINI_RETRY_DELAYS_MS = [5000, 20000, 60000];


function getConfig_() {
  var props = PropertiesService.getScriptProperties();
  var apiKey = props.getProperty("GEMINI_API_KEY");
  if (!apiKey) {
    throw new Error(
      "GEMINI_API_KEY is not set. Open File → Project properties → " +
      "Script properties and add row GEMINI_API_KEY=<your key>."
    );
  }
  return {
    apiKey: apiKey,
    projectId:        props.getProperty("FIREBASE_PROJECT_ID") || DEFAULT_FIREBASE_PROJECT_ID,
    folderId:         props.getProperty("FOLDER_ID")           || DEFAULT_FOLDER_ID,
    summaryFolderId:  props.getProperty("SUMMARY_FOLDER_ID")   || null,
    timezone:         props.getProperty("SUMMARY_TZ")          || "Europe/Moscow"
  };
}


/**
 * Compute a 768-dim embedding for `text` using gemini-embedding-001.
 * Returns array of floats, or null on failure (we still save the doc — backfill
 * can fill missing embeddings later).
 */
function computeEmbedding_(text, cfg) {
  if (!text || !text.trim()) return null;
  var url = "https://generativelanguage.googleapis.com/v1beta/models/" +
            "gemini-embedding-001:embedContent?key=" + encodeURIComponent(cfg.apiKey);
  var payload = {
    "content": { "parts": [{ "text": String(text) }] },
    "outputDimensionality": 768
  };
  try {
    var resp = UrlFetchApp.fetch(url, {
      "method": "post",
      "contentType": "application/json",
      "payload": JSON.stringify(payload),
      "muteHttpExceptions": true
    });
    var code = resp.getResponseCode();
    if (code !== 200) {
      console.warn("embedContent HTTP " + code + ": " + resp.getContentText().slice(0, 300));
      return null;
    }
    var data = JSON.parse(resp.getContentText());
    var vals = data && data.embedding && data.embedding.values;
    if (Array.isArray(vals) && vals.length) {
      console.log("embedding ok: dims=" + vals.length + " text_len=" + text.length);
      return vals;
    }
    console.warn("embedContent returned empty vector for text_len=" + text.length);
    return null;
  } catch (e) {
    console.warn("embedContent failed: " + e);
    return null;
  }
}


/**
 * Main trigger entry. Run on a time-driven schedule.
 */
function processNewLogs() {
  try {
    var cfg = getConfig_();
    var folder = DriveApp.getFolderById(cfg.folderId);
    var files = folder.getFiles();
    var startedAt = Date.now();
    var processedThisRun = 0;

    while (files.hasNext()) {
      if (Date.now() - startedAt > MAX_RUNTIME_MS) {
        console.log("Runtime budget reached after " + processedThisRun +
                    " files. Remaining unprocessed files will be picked up " +
                    "by the next trigger.");
        break;
      }

      var file = files.next();
      var desc = file.getDescription() || "";

      var isText = file.getMimeType() === "text/plain";
      var notProcessed = desc.indexOf("processed") === -1;
      if (isText && (FORCE_REPROCESS || notProcessed)) {
        if (processedThisRun > 0) {
          // Spread requests across time to stay under Gemini RPM cap.
          Utilities.sleep(INTER_FILE_PAUSE_MS);
        }
        console.log(">>> Processing: " + file.getName());
        var ok = runMining_(file, cfg);
        if (ok) {
          file.setDescription("processed_at_" + new Date().toISOString());
          console.log("<<< Done: " + file.getName());
        } else {
          console.warn(
            file.getName() + ": Gemini call failed after retries. " +
            "Not marking processed — will retry on next run."
          );
        }
        processedThisRun++;
      }
    }
  } catch (e) {
    console.error("Critical: " + e.toString());
  }
}


function runMining_(file, cfg) {
  try {
    var content = file.getBlob().getDataAsString();
    var sourceDateISO = parseFileNameDate_(file.getName());
    var modelIdx = _modelCallCounter % MODELS.length;
    _modelCallCounter++;

    var systemPrompt =
      "Ты — главный архитектор данных MultySpeak. Фильтрация и маршрутизация опыта из чатов.\n" +
      "Преврати лог в JSON: {\"patterns\": [...]}.\n\n" +
      "ОБЯЗАТЕЛЬНЫЕ ПОЛЯ КАЖДОГО ЭЛЕМЕНТА:\n" +
      "- title: на английском (универсальный ключ).\n" +
      "- country: ISO 3166-1 alpha-2 в нижнем регистре (br, id, lk, vn, tr, и т.д. " +
      "по стандарту). Если pattern универсальный и не привязан к одной стране " +
      "(общий лайфхак для путешественников, советы по технике, и т.п.) — " +
      "укажи \"any\".\n" +
      "- routing: одна из строк. ВЫБИРАЙ ПО ПРАВИЛАМ:\n" +
      "    * \"both\" — есть И живая история/байка/контекст, И полезный сухой " +
      "факт (цифры, цены, инструкция, ссылка). ЭТО ДЕФОЛТ — большинство " +
      "интересных кейсов сюда. Если сомневаешься — ставь \"both\".\n" +
      "    * \"assistant_only\" — голый сухой факт без живой истории. Например: " +
      "контакт чиновника, точная цена, шаг бюрократической процедуры, " +
      "название документа. ИИ-помощнику пригодится, но публиковать в канал " +
      "скучно.\n" +
      "    * \"channel_only\" — живая байка/мем/локальный колорит без " +
      "извлекаемого факта. Например: смешная история без конкретики, " +
      "наблюдение про менталитет, культурный анекдот. В канал интересно, " +
      "ассистенту нечего из этого вытащить.\n" +
      "- tag: на английском (Finance, Safety, Bureaucracy, Travel и т.п.).\n" +
      "- target_languages: массив ISO 639-1 кодов языков на которые история " +
      "имеет смысл переводиться. По умолчанию [\"ru\"] для русско-локальных " +
      "историй (российские лайфхаки, нюансы для русскоязычной диаспоры). " +
      "Универсальные истории (карнавал, погода, кухня, общая туристика) — " +
      "перечисли все уместные: например [\"ru\",\"en\",\"es\",\"pt\"]. " +
      "Только если routing == \"both\" или \"channel_only\".\n" +
      "- human_story: ИСТОРИИ И ХАКИ ДЛЯ КАНАЛА. СТРОГО НА РУССКОМ ЯЗЫКЕ. " +
      "Пиши сочно, живо, с лёгкой иронией. Сделай это интересной историей для " +
      "канала. Только если routing == \"both\" или \"channel_only\".\n" +
      "- ai_lesson: ИНСТРУКЦИЯ ДЛЯ ИИ-ПОМОЩНИКА. СТРОГО НА АНГЛИЙСКОМ ЯЗЫКЕ. " +
      "Сухие, точные факты и цифры без эмоций. Только если routing == \"both\" " +
      "или \"assistant_only\".\n\n" +
      "ФИЛЬТРАЦИЯ:\n" +
      "- Игнорируй слухи, пустой трёп, спам и сообщения про спамеров.\n" +
      "- Игнорируй pattern если в логе нет конкретики — не выдумывай.\n";

    var payload = {
      "contents": [{ "parts": [{ "text": "Текст лога:\n" + content }] }],
      "systemInstruction": { "parts": [{ "text": systemPrompt }] },
      "generationConfig": { "responseMimeType": "application/json" }
    };

    var response = fetchGeminiWithRetry_(payload, file.getName(), cfg, modelIdx);
    if (!response) return false;

    var result = JSON.parse(response.getContentText());
    if (!result.candidates || !result.candidates[0] || !result.candidates[0].content) {
      console.error("Gemini empty/blocked response: " +
                    JSON.stringify(result).slice(0, 500));
      return false;
    }

    var rawText = result.candidates[0].content.parts[0].text;
    var clean = rawText.replace(/```json|```/g, "").trim();
    var data = JSON.parse(clean);
    var patterns = data.patterns || data;

    if (!Array.isArray(patterns)) {
      console.error("Gemini returned no patterns array.");
      return false;
    }

    var saved = 0;
    var attempted = 0;
    var wisdomLines = [];
    var storyLines = [];
    patterns.forEach(function(p, idx) {
      if (!p || typeof p !== "object") return;
      var routing = p.routing || "both";

      if (routing === "both" || routing === "assistant_only") {
        attempted++;
        if (saveToFirestore_(p, COLLECTION_AI, false, file.getId(), idx, cfg, sourceDateISO)) {
          saved++;
          wisdomLines.push(formatSummaryLine_(p, "wisdom", file.getName()));
        }
      }
      if ((routing === "both" || routing === "channel_only") && p.human_story) {
        attempted++;
        if (saveToFirestore_(p, COLLECTION_TG, true, file.getId(), idx, cfg, sourceDateISO)) {
          saved++;
          storyLines.push(formatSummaryLine_(p, "story", file.getName()));
        }
      }
    });

    // Flush per-file so a runtime-budget exit doesn't lose what we already
    // wrote to Firestore. One Drive read+write per file per kind.
    if (wisdomLines.length) appendDailyLog_(cfg, "wisdom", wisdomLines);
    if (storyLines.length)  appendDailyLog_(cfg, "stories", storyLines);

    if (attempted === 0) {
      console.log("No patterns to save from " + file.getName() +
                  " (" + patterns.length + " raw patterns, all filtered out).");
    } else {
      console.log("Saved " + saved + " of " + attempted + " patterns from " + file.getName());
    }

    // File is "processed" if Gemini gave us a valid response, even if patterns
    // are empty (nothing in the chat worth saving) or all filtered. Don't
    // retry forever on legitimately-uninteresting files.
    // Real save failures (saved < attempted) are logged but still mark the
    // file processed — re-process via FORCE_REPROCESS=true if needed.
    if (saved < attempted) {
      console.warn(file.getName() + ": " + (attempted - saved) +
                   " of " + attempted + " saves failed (see Firestore errors above).");
    }
    return true;

  } catch (e) {
    console.error("runMining_ error: " + e.toString());
    return false;
  }
}


/**
 * Idempotent Firestore writer.
 *
 * Doc ID = SHA1(fileId + ":" + idx + ":" + collection).slice(0, 24).
 * Re-running on the same patterns produces the same ID → server replies
 * 409 ALREADY_EXISTS, we treat as success → no duplicates.
 *
 * Returns true on 200 OK or 409 ALREADY_EXISTS, false on real failure.
 */
function saveToFirestore_(item, collectionName, isPost, fileId, idx, cfg, sourceDateISO) {
  var seed = fileId + ":" + idx + ":" + collectionName;
  var hex = sha1Hex_(seed).slice(0, 24);

  var url = "https://firestore.googleapis.com/v1/projects/" + cfg.projectId +
            "/databases/default/documents/" + collectionName +
            "?documentId=" + hex;

  var fields = {
    "title":        { "stringValue": item.title || "Untitled" },
    "country":      { "stringValue": String(item.country || "unknown").toLowerCase() },
    "tag":          { "stringValue": item.tag || "General" },
    "createdAt":    { "timestampValue": new Date().toISOString() },
    "sourceFileId": { "stringValue": fileId },
    "sourceIdx":    { "integerValue": idx }
  };
  // sourceDate = day the source chat log is from (parsed from filename).
  // Lets bot show "last confirmed: YYYY-MM-DD" and flag stale facts.
  // Omitted if filename doesn't match YYYY-MM-DD_*.txt pattern.
  if (sourceDateISO) {
    fields["sourceDate"] = { "timestampValue": sourceDateISO };
  }

  if (isPost) {
    fields["content"] = { "stringValue": item.human_story || "" };
    // Languages this story is meaningful in. Default to russian if Gemini
    // didn't provide. Bot matches against channel.language at post time.
    var langs = Array.isArray(item.target_languages) && item.target_languages.length
      ? item.target_languages : ["ru"];
    fields["target_languages"] = {
      "arrayValue": {
        "values": langs.map(function(l) { return { "stringValue": String(l).toLowerCase() }; })
      }
    };
    // postedTo starts empty. Bot enriches per-channel as it processes:
    //   postedTo.<channel>: {
    //     posted: bool, scheduled_at: ts, posted_at: ts|null,
    //     language: str, text: str (actual posted version after translation)
    //   }
    // Kept empty here so Apps Script doesn't need to know channel inventory.
    fields["postedTo"] = { "mapValue": { "fields": {} } };
  } else {
    var instruction = item.ai_lesson || "";
    fields["instruction"] = { "stringValue": instruction };
    // Compute embedding for wisdom — without this МОЗГ's find_nearest can't
    // retrieve the entry. Backfill exists as a safety net, but doing it inline
    // means new wisdom is searchable immediately.
    var emb = computeEmbedding_(instruction, cfg);
    if (emb && emb.length) {
      // Firestore Vector type — required for find_nearest. Plain arrayValue
      // is stored but is invisible to vector search (same trap we hit with
      // wikivoyage_base earlier). The {__type__: __vector__, value: [...]}
      // map is the REST-API equivalent of Python's google.cloud Vector().
      fields["embedding"] = {
        "mapValue": {
          "fields": {
            "__type__": { "stringValue": "__vector__" },
            "value": {
              "arrayValue": {
                "values": emb.map(function(v) { return { "doubleValue": v }; })
              }
            }
          }
        }
      };
    }
  }

  var response = UrlFetchApp.fetch(url, {
    "method": "post",
    "contentType": "application/json",
    "headers": { "Authorization": "Bearer " + ScriptApp.getOAuthToken() },
    "payload": JSON.stringify({ "fields": fields }),
    "muteHttpExceptions": true
  });

  var code = response.getResponseCode();
  if (code === 200) return true;
  if (code === 409) {
    // ALREADY_EXISTS — same deterministic doc id from earlier run. Idempotent.
    return true;
  }
  console.error("Firestore " + collectionName + " HTTP " + code + ": " +
                response.getContentText().slice(0, 500));
  return false;
}


/**
 * Gemini POST with retry on transient failures (503/429).
 *
 * Returns response object on HTTP 200, or null after exhausting retries.
 * 503 ("model overloaded") and 429 ("rate limit") are very common on the
 * free tier — backoff lets capacity recover. Other HTTP errors (4xx, 5xx)
 * are not retried, just logged.
 */
function fetchGeminiWithRetry_(payload, fileName, cfg, startModelIdx) {
  // 429 (rate limit) → switch model immediately, don't wait.
  // 503/500 → backoff, same model (server-side overload, model change won't help).
  var maxAttempts = GEMINI_RETRY_DELAYS_MS.length + 1;
  var modelIdx = startModelIdx;
  var triedModels = {};
  for (var attempt = 1; attempt <= maxAttempts; attempt++) {
    var model = MODELS[modelIdx % MODELS.length];
    triedModels[model] = true;
    var url = "https://generativelanguage.googleapis.com/v1beta/models/" + model +
              ":generateContent?key=" + cfg.apiKey;
    console.log(fileName + " → " + model + " (attempt " + attempt + ")");
    var response = UrlFetchApp.fetch(url, {
      "method": "post",
      "contentType": "application/json",
      "payload": JSON.stringify(payload),
      "muteHttpExceptions": true
    });
    var code = response.getResponseCode();
    if (code === 200) return response;

    var body = response.getContentText().slice(0, 500);

    var transient = (code === 503 || code === 429 || code === 500);

    // Any transient error → try the other model FIRST before sleeping.
    // 429 = our RPM bucket, 503 = Google's capacity for this model — both
    // are likely solved by switching to the sibling model with its own
    // bucket and capacity pool.
    if (transient && Object.keys(triedModels).length < MODELS.length) {
      modelIdx = (modelIdx + 1) % MODELS.length;
      console.warn(fileName + ": HTTP " + code + " on " + model +
                   " — switching to sibling model. " + body);
      continue;
    }

    // Both models tried and still failing — back off and retry.
    if (transient && attempt < maxAttempts) {
      var waitMs = GEMINI_RETRY_DELAYS_MS[attempt - 1];
      console.warn(fileName + ": HTTP " + code + ", both models busy, " +
                   "backing off " + (waitMs / 1000) + "s. " + body);
      Utilities.sleep(waitMs);
      // Reset triedModels so after the sleep we try fresh.
      triedModels = {};
      continue;
    }

    console.error(fileName + ": Gemini HTTP " + code + " on " + model +
                  " (attempt " + attempt + "/" + maxAttempts + "): " + body);
    return null;
  }
  return null;
}


/**
 * Extract source-log date from filename like "2026-05-07_br_chatforum.txt".
 *
 * Returns ISO 8601 timestamp at noon UTC of that date, or null if the
 * filename doesn't match the expected YYYY-MM-DD_ prefix. Noon UTC chosen
 * to avoid timezone-edge-flip when bot renders the date locally.
 */
function parseFileNameDate_(fileName) {
  var m = fileName.match(/^(\d{4})-(\d{2})-(\d{2})_/);
  if (!m) return null;
  // Build ISO directly — Date parsing varies, this is safer.
  return m[1] + "-" + m[2] + "-" + m[3] + "T12:00:00.000Z";
}


/**
 * Format a single summary line for daily TXT log.
 *
 * kind="wisdom" → first sentence of ai_lesson (EN, dry fact)
 * kind="story"  → first sentence of human_story (RU, narrative)
 *
 * Line shape: [country/tag] title :: first sentence, capped at ~250 chars.
 */
function formatSummaryLine_(p, kind, sourceFileName) {
  var country = String(p.country || "??").toLowerCase();
  var tag     = p.tag   || "General";
  var title   = p.title || "Untitled";
  var body    = kind === "wisdom" ? (p.ai_lesson || "") : (p.human_story || "");
  // Split on first sentence terminator. Fall back to first 200 chars.
  var firstSentence = body.split(/[.!?\n]/)[0].trim();
  if (firstSentence.length > 200) firstSentence = firstSentence.slice(0, 200) + "…";
  return "[" + country + "/" + tag + "] " + title + " :: " + firstSentence +
         "  (src: " + sourceFileName + ")";
}


/**
 * Append lines to today's summary TXT in Drive.
 *
 * File naming: YYYY-MM-DD_<kind>.txt in SUMMARY_FOLDER_ID. Date = current
 * run date in configured timezone, not source-log date. So a FORCE_REPROCESS
 * burst lands in one summary file showing "what was mined today".
 *
 * Created on first write of the day with a header. Subsequent calls read +
 * append + write back (Apps Script has no native append for plaintext).
 *
 * Silently no-op if SUMMARY_FOLDER_ID not configured — pipeline still works.
 */
function appendDailyLog_(cfg, kind, lines) {
  if (!cfg.summaryFolderId) return;
  if (!lines || !lines.length) return;

  var date = Utilities.formatDate(new Date(), cfg.timezone, "yyyy-MM-dd");
  var fileName = date + "_" + kind + ".txt";
  var folder;
  try {
    folder = DriveApp.getFolderById(cfg.summaryFolderId);
  } catch (e) {
    console.warn("SUMMARY_FOLDER_ID invalid (" + cfg.summaryFolderId +
                 "): " + e.toString() + ". Skipping daily log.");
    return;
  }

  var existing = folder.getFilesByName(fileName);
  var file, content;
  if (existing.hasNext()) {
    file = existing.next();
    content = file.getBlob().getDataAsString();
  } else {
    var header = "# Mining summary " + date + " — " +
                 (kind === "wisdom" ? "wisdom_base (assistant data)"
                                    : "telegram_queue (channel stories)") +
                 "\n# Each line: [country/tag] title :: first sentence  (src: file)\n\n";
    file = folder.createFile(fileName, header, "text/plain");
    content = header;
  }

  content += lines.join("\n") + "\n";
  file.setContent(content);
}


/**
 * Hex SHA1 of a string.
 */
function sha1Hex_(s) {
  var bytes = Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_1, s);
  var hex = "";
  for (var i = 0; i < bytes.length; i++) {
    var b = bytes[i] < 0 ? bytes[i] + 256 : bytes[i];
    hex += (b < 16 ? "0" : "") + b.toString(16);
  }
  return hex;
}


// --- Manual utilities (not triggered automatically) ---


/**
 * Force re-auth of all OAuth scopes the pipeline needs.
 *
 * Run this manually from the editor whenever processNewLogs fails with
 * "You do not have permission to call ...". Apps Script caches granted
 * scopes per known function — a new function with explicit calls to every
 * required scope forces a fresh consent screen.
 *
 * Touches:
 *   - Drive read  (DriveApp.getRootFolder)
 *   - Drive write (createFile + setDescription + setTrashed)
 *   - External HTTP (UrlFetchApp.fetch)
 *   - OAuth token  (ScriptApp.getOAuthToken — datastore / cloud-platform)
 *
 * A readonly-only probe like getName() will grant just drive.readonly,
 * which is NOT enough for setDescription on processed files — hence the
 * write probe.
 */
function forceAuth() {
  var name = DriveApp.getRootFolder().getName();
  console.log("Drive read OK, root: " + name);

  var probe = DriveApp.createFile("_auth_probe.txt", "probe", "text/plain");
  probe.setDescription("test");
  probe.setTrashed(true);
  console.log("Drive write OK");

  var resp = UrlFetchApp.fetch("https://www.google.com/generate_204", {
    "muteHttpExceptions": true
  });
  console.log("UrlFetch OK, status: " + resp.getResponseCode());

  var token = ScriptApp.getOAuthToken();
  console.log("OAuth token OK, len: " + token.length);
}


/**
 * Strip "processed_at_..." markers from every file in the input folder.
 *
 * After a Firestore wipe, files still carry processed-markers from the
 * previous mining run. With FORCE_REPROCESS=false the pipeline would skip
 * them entirely (nothing left in DB). With FORCE_REPROCESS=true the
 * pipeline cycles forever on the first batch (runtime budget exits, next
 * trigger restarts from file #1, never advances past 5 min worth of files).
 *
 * Run this once after a wipe, then leave FORCE_REPROCESS=false. Triggers
 * will then advance batch-by-batch through the unmarked files exactly
 * like a fresh upload.
 */
function clearAllMarkers() {
  var cfg = getConfig_();
  var folder = DriveApp.getFolderById(cfg.folderId);
  var files = folder.getFiles();
  var count = 0;
  while (files.hasNext()) {
    var f = files.next();
    if (f.getDescription()) {
      f.setDescription("");
      count++;
    }
  }
  console.log("Cleared markers on " + count + " files");
}
