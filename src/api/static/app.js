/**
 * Aviator ML Research Dashboard — Live Telemetry & Causal Prediction Engine
 */

let liveWs = null;
let liveReconnectTimer = null;
let currentMode = "live"; // "live" | "csv"
const AVIATOR_OBSERVER_SNIPPET = `/**
 * Aviator Live Game Observer (Read-Only) — Deduplicated Telemetry Bridge
 *
 * For: aviator-next.spribegaming.com (or any Aviator game iframe)
 * Exposes:
 *   - window.sendAviatorTestFrame(multiplier, roundId)
 *   - window.getAviatorDiagnostics()
 *   - window.scanAviatorWebSockets()
 *   - window.connectAviatorBridge()
 *   - window.getAviatorBridgeStatus()
 *
 * SAFETY AUDIT:
 * - Strictly passive read-only event listener.
 * - ZERO betting, zero wagering, zero balance interactions.
 * - Does NOT alter incoming or outgoing game frames.
 * - De-duplicates frame processing per socket and message.
 */

(function (root) {
  "use strict";

  var BRIDGE_PAGE_URL = "http://127.0.0.1:8000/bridge.html";
  var globalObj = typeof window !== "undefined" ? window : (typeof globalThis !== "undefined" ? globalThis : root);

  // Clear existing timers from prior snippet injections
  if (globalObj.__aviator_scan_interval) {
    try { clearInterval(globalObj.__aviator_scan_interval); } catch (e) {}
    globalObj.__aviator_scan_interval = null;
  }
  if (globalObj.__aviator_ping_interval) {
    try { clearInterval(globalObj.__aviator_ping_interval); } catch (e) {}
    globalObj.__aviator_ping_interval = null;
  }

  // Preserve global state across re-runs to prevent duplicate socket detections
  if (!globalObj.__aviator_hooked_sockets) {
    globalObj.__aviator_hooked_sockets = typeof WeakSet !== "undefined" ? new WeakSet() : { has: function () { return false; }, add: function () {} };
  }
  var hookedSockets = globalObj.__aviator_hooked_sockets;

  if (!globalObj.__aviator_discovered_sockets) {
    globalObj.__aviator_discovered_sockets = [];
  }
  var discoveredSockets = globalObj.__aviator_discovered_sockets;

  // Internal state
  var bridgeWindow = null;
  var isBridgeConnected = false;
  var pendingQueue = [];
  var framesSentCount = 0;
  var rawFramesCapturedCount = 0;
  var lastRawFrameSnippet = null;
  var isIframe = typeof window !== "undefined" && window.self !== window.top;
  var hostName = typeof window !== "undefined" && window.location ? window.location.host : "unknown";

  // De-duplication and rate-limiting buffers
  var recentSignatures = [];
  var MAX_RECENT_SIGS = 100;
  var lastConsoleLogTime = 0;
  var suppressedLogCount = 0;

  function isDuplicateFrame(sig) {
    var now = Date.now();
    recentSignatures = recentSignatures.filter(function (entry) {
      return (now - entry.ts) < 2000;
    });
    for (var i = 0; i < recentSignatures.length; i++) {
      if (recentSignatures[i].sig === sig) {
        return true;
      }
    }
    recentSignatures.push({ sig: sig, ts: now });
    if (recentSignatures.length > MAX_RECENT_SIGS) {
      recentSignatures.shift();
    }
    return false;
  }

  // =========================================================================
  // 1. BRIDGE CONNECTION & FORWARDING
  // =========================================================================
  function findExistingBridge() {
    if (bridgeWindow && !bridgeWindow.closed) return bridgeWindow;
    try { if (window.opener && !window.opener.closed) return window.opener; } catch (e) {}
    try { if (window.top && window.top.opener && !window.top.opener.closed) return window.top.opener; } catch (e) {}
    try { if (window.parent && window.parent.opener && !window.parent.opener.closed) return window.parent.opener; } catch (e) {}
    return null;
  }

  function openBridgeWindow() {
    var existing = findExistingBridge();
    if (existing) {
      bridgeWindow = existing;
      isBridgeConnected = true;
      return existing;
    }
    try {
      bridgeWindow = window.open(
        BRIDGE_PAGE_URL,
        "AviatorTelemetryBridge",
        "width=540,height=560,menubar=no,toolbar=no,location=no,status=no"
      );
      if (bridgeWindow) {
        console.log(
          "%c[Aviator ML Observer]%c 🚀 Telemetry Bridge companion window opened.",
          "color: #10b981; font-weight: bold;",
          "color: inherit;"
        );
      }
    } catch (err) {
      console.warn("[Aviator ML Observer] Could not open bridge window (popup blocked):", err);
    }
    return bridgeWindow;
  }

  function postToBridge(payload) {
    var target = findExistingBridge() || bridgeWindow;
    var messageObj = {
      __aviator_telemetry: true,
      source: "aviator_observer",
      context: isIframe ? "iframe" : "top",
      host: hostName,
      timestamp: new Date().toISOString(),
      raw_frame: typeof payload === "string" ? payload : JSON.stringify(payload)
    };

    if (target && !target.closed) {
      try {
        target.postMessage(messageObj, "*");
        framesSentCount++;
        return true;
      } catch (err) {
        console.warn("[Aviator ML Observer] Error posting to bridge:", err);
      }
    }

    pendingQueue.push(messageObj);
    if (pendingQueue.length > 150) pendingQueue.shift();
    return false;
  }

  function flushPendingQueue() {
    var target = findExistingBridge() || bridgeWindow;
    if (!target || target.closed) return;
    while (pendingQueue.length > 0) {
      var item = pendingQueue.shift();
      try {
        target.postMessage(item, "*");
        framesSentCount++;
      } catch (e) {
        break;
      }
    }
  }

  if (typeof window !== "undefined") {
    window.addEventListener("message", function (event) {
      var data = event.data;
      if (!data) return;
      if (data.type === "bridge_handshake_ack" || data.type === "bridge_ready" || data.type === "bridge_ack") {
        isBridgeConnected = true;
        if (event.source && (!bridgeWindow || bridgeWindow.closed)) {
          bridgeWindow = event.source;
        }
        flushPendingQueue();
      }
    });
  }

  // =========================================================================
  // 2. BINARY FRAME FORMATTER (Requirement 5 & 6)
  // =========================================================================
  function arrayBufferToBase64(buffer) {
    if (!buffer) return null;
    try {
      var bytes = new Uint8Array(buffer);
      var binary = "";
      var len = bytes.byteLength;
      for (var i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
      }
      return typeof btoa !== "undefined" ? btoa(binary) : null;
    } catch (e) {
      return null;
    }
  }

  function formatBinaryData(data) {
    var byteLength = 0;
    var bytes = null;
    if (data instanceof ArrayBuffer) {
      byteLength = data.byteLength;
      bytes = new Uint8Array(data);
    } else if (typeof ArrayBuffer !== "undefined" && ArrayBuffer.isView && ArrayBuffer.isView(data)) {
      byteLength = data.byteLength;
      bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    } else if (typeof Blob !== "undefined" && data instanceof Blob) {
      byteLength = data.size;
    }

    var safePreview = "";
    if (bytes) {
      var sliceLen = Math.min(16, bytes.length);
      var hexParts = [];
      for (var i = 0; i < sliceLen; i++) {
        var h = bytes[i].toString(16);
        hexParts.push("0x" + (h.length === 1 ? "0" + h : h));
      }
      safePreview = hexParts.join(" ") + (byteLength > 16 ? " ... (" + byteLength + " bytes)" : "");
    } else {
      safePreview = "(" + byteLength + " bytes)";
    }

    // Try safe UTF-8 decode
    var decodedText = null;
    if (bytes && typeof TextDecoder !== "undefined") {
      try {
        var txt = new TextDecoder("utf-8", { fatal: false }).decode(bytes);
        if (txt && (txt.startsWith("{") || txt.startsWith("[") || txt.startsWith("a["))) {
          decodedText = txt;
        } else if (txt) {
          // Check for embedded JSON payload in Protobuf / Centrifugo message
          var firstBrace = txt.indexOf("{");
          var lastBrace = txt.lastIndexOf("}");
          if (firstBrace !== -1 && lastBrace > firstBrace) {
            var candidate = txt.slice(firstBrace, lastBrace + 1);
            try {
              var parsed = JSON.parse(candidate);
              if (parsed && (parsed.type || parsed.stage || parsed.multiplier || parsed.round_id || parsed.roundId)) {
                decodedText = candidate;
              }
            } catch (e) {}
          }
        }
      } catch (e) {}
    }

    var b64 = bytes ? arrayBufferToBase64(bytes.buffer) : null;

    return {
      byteLength: byteLength,
      encoding: decodedText ? "text/utf-8-json" : (bytes ? "binary/protobuf" : "binary/bytes"),
      safePreview: safePreview,
      decodedText: decodedText,
      payloadB64: b64
    };
  }

  // =========================================================================
  // 3. FRAME PROCESSING & DISPATCH WITH DE-DUPLICATION
  // =========================================================================
  function handleIncomingMessage(data, socketUrl) {
    if (typeof Blob !== "undefined" && data instanceof Blob) {
      if (typeof data.text === "function") {
        data.text().then(function (text) {
          if (text && (text.startsWith("{") || text.startsWith("[") || text.startsWith("a["))) {
            handleIncomingMessage(text, socketUrl);
          } else if (typeof data.arrayBuffer === "function") {
            data.arrayBuffer().then(function (buf) {
              handleIncomingMessage(buf, socketUrl);
            });
          }
        }).catch(function () {
          if (typeof data.arrayBuffer === "function") {
            data.arrayBuffer().then(function (buf) {
              handleIncomingMessage(buf, socketUrl);
            });
          }
        });
        return;
      }
    }

    var rawText = null;
    var isBinary = false;
    var binInfo = null;

    if (typeof data === "string") {
      rawText = data;
    } else if (typeof Blob !== "undefined" && data instanceof Blob) {
      binInfo = formatBinaryData(data);
      if (binInfo.decodedText) {
        rawText = binInfo.decodedText;
      } else {
        isBinary = true;
      }
    } else if (data instanceof ArrayBuffer || (typeof ArrayBuffer !== "undefined" && ArrayBuffer.isView && ArrayBuffer.isView(data))) {
      binInfo = formatBinaryData(data);
      if (binInfo.decodedText) {
        rawText = binInfo.decodedText;
      } else {
        isBinary = true;
      }
    }

    // De-duplication key
    var sig = isBinary
      ? (socketUrl + ":bin:" + binInfo.byteLength + ":" + binInfo.safePreview)
      : (socketUrl + ":" + (rawText || "").slice(0, 100));

    if (isDuplicateFrame(sig)) {
      return;
    }

    rawFramesCapturedCount++;

    // Safe handling for binary frames (Requirement 5 & 6)
    if (isBinary) {
      lastRawFrameSnippet = "Binary (" + binInfo.byteLength + "B): " + binInfo.safePreview;
      var now = Date.now();
      if (now - lastConsoleLogTime >= 1000) {
        console.log(
          "%c[Aviator ML Observer] 📦 BINARY FRAME #" + rawFramesCapturedCount + ":%c " + binInfo.safePreview,
          "color: #38bdf8; font-weight: bold;",
          "color: inherit;"
        );
        lastConsoleLogTime = now;
      }
      postToBridge({
        type: "binary_frame",
        socket_url: socketUrl || "unknown",
        byte_length: binInfo.byteLength,
        encoding: binInfo.encoding,
        safe_preview: binInfo.safePreview,
        payload_b64: binInfo.payloadB64,
        frame_index: rawFramesCapturedCount,
        timestamp: new Date().toISOString()
      });
      return;
    }

    if (!rawText) return;

    lastRawFrameSnippet = rawText.length > 250 ? rawText.slice(0, 250) + "..." : rawText;

    // Rate-limited console logging for game frames (Requirement 7)
    var isHeartbeat = rawText === "h" || rawText === "o" || rawText === "3" || rawText === "[]";
    var nowTs = Date.now();
    if (!isHeartbeat) {
      if (nowTs - lastConsoleLogTime >= 1000) {
        var extra = suppressedLogCount > 0 ? " (+" + suppressedLogCount + " throttled)" : "";
        console.log(
          "%c[Aviator ML Observer] 🎮 REAL GAME FRAME #" + rawFramesCapturedCount + extra + ":%c " + lastRawFrameSnippet,
          "color: #10b981; font-weight: bold; background: #064e3b; padding: 2px 5px; border-radius: 4px;",
          "color: inherit;"
        );
        lastConsoleLogTime = nowTs;
        suppressedLogCount = 0;
      } else {
        suppressedLogCount++;
      }
    }

    // Forward frame to Bridge
    postToBridge({
      type: "game_frame",
      raw_frame: rawText,
      socket_url: socketUrl || "unknown",
      frame_index: rawFramesCapturedCount,
      timestamp: new Date().toISOString()
    });
  }

  // =========================================================================
  // 4. WEBSOCKET HOOKING: SINGLE LISTENER PER SOCKET INSTANCE
  // =========================================================================
  function hookSocketInstance(ws, detectionSource) {
    if (!ws) return;
    try {
      if (hookedSockets.has(ws) || ws.__aviator_hooked) return;
      hookedSockets.add(ws);
      ws.__aviator_hooked = true;
    } catch (e) {}

    var url = "unknown";
    try { url = ws.url || "ws://unknown"; } catch (e) {}
    var rState = -1;
    try { rState = ws.readyState; } catch (e) {}

    var alreadyKnown = false;
    for (var i = 0; i < discoveredSockets.length; i++) {
      if (discoveredSockets[i].url === url) {
        alreadyKnown = true;
        break;
      }
    }
    if (!alreadyKnown) {
      var socketRecord = {
        url: url,
        readyState: rState,
        detectedVia: detectionSource,
        timestamp: new Date().toISOString()
      };
      discoveredSockets.push(socketRecord);

      console.log(
        "%c[Aviator ML Observer]%c 🔌 Hooked active WebSocket (" + detectionSource + "): " + url + " (readyState: " + rState + ")",
        "color: #10b981; font-weight: bold;",
        "color: inherit;"
      );

      postToBridge({
        type: "socket_detected",
        url: url,
        ready_state: rState,
        detection_source: detectionSource,
        timestamp: new Date().toISOString()
      });
    }

    // Attach exactly ONE message listener to this socket instance
    try {
      ws.addEventListener("message", function (event) {
        if (!event) return;
        if (event.__aviator_handled) return;
        try { event.__aviator_handled = true; } catch (e) {}
        handleIncomingMessage(event.data, url);
      });
    } catch (e) {}
  }

  // Hook WebSocket.prototype.send (discovers active socket when it transmits)
  try {
    if (typeof WebSocket !== "undefined" && WebSocket.prototype && !WebSocket.prototype.__aviator_send_hooked) {
      var originalWSSend = WebSocket.prototype.send;
      WebSocket.prototype.send = function () {
        hookSocketInstance(this, "prototype_send");
        return originalWSSend.apply(this, arguments);
      };
      WebSocket.prototype.__aviator_send_hooked = true;
    }
  } catch (err) {}

  // Hook WebSocket.prototype.addEventListener (discovers socket on listener registration)
  try {
    if (typeof WebSocket !== "undefined" && WebSocket.prototype && !WebSocket.prototype.__aviator_ael_hooked) {
      var originalWSAEL = WebSocket.prototype.addEventListener;
      WebSocket.prototype.addEventListener = function (type, listener, options) {
        hookSocketInstance(this, "prototype_addEventListener");
        return originalWSAEL.apply(this, arguments);
      };
      WebSocket.prototype.__aviator_ael_hooked = true;
    }
  } catch (err) {}

  // Hook WebSocket.prototype.onmessage setter (discovers socket on assignment)
  try {
    if (typeof WebSocket !== "undefined" && WebSocket.prototype) {
      var originalOnMessageDesc = Object.getOwnPropertyDescriptor(WebSocket.prototype, "onmessage");
      if (originalOnMessageDesc && originalOnMessageDesc.set && !originalOnMessageDesc.__aviator_desc_hooked) {
        Object.defineProperty(WebSocket.prototype, "onmessage", {
          set: function (fn) {
            hookSocketInstance(this, "prototype_onmessage_setter");
            return originalOnMessageDesc.set.call(this, fn);
          },
          get: function () {
            return originalOnMessageDesc.get.call(this);
          },
          configurable: true,
          enumerable: true
        });
        WebSocket.prototype.__aviator_desc_hooked = true;
      }
    }
  } catch (err) {}

  // Hook global WebSocket constructor for future instantiations
  function patchWSConstructor(win) {
    if (!win || win.__aviator_ws_patched) return;
    try {
      var OriginalWS = win.WebSocket;
      if (!OriginalWS) return;

      win.WebSocket = function () {
        var args = Array.prototype.slice.call(arguments);
        var ws = new (Function.prototype.bind.apply(OriginalWS, [null].concat(args)))();
        hookSocketInstance(ws, "constructor_new");
        return ws;
      };

      win.WebSocket.prototype = OriginalWS.prototype;
      win.__aviator_ws_patched = true;
    } catch (err) {}
  }
  patchWSConstructor(globalObj);

  // =========================================================================
  // 5. SCAN FOR ALREADY-EXISTING WEBSOCKETS IN MEMORY & CHILD IFRAMES
  // =========================================================================
  function scanForExistingSockets() {
    var countBefore = discoveredSockets.length;
    var scannedObjects = typeof Set !== "undefined" ? new Set() : { has: function () { return false; }, add: function () {} };

    function walk(target, depth, path) {
      if (!target || depth > 3 || scannedObjects.has(target)) return;
      try { scannedObjects.add(target); } catch (e) {}

      try {
        var props = Object.getOwnPropertyNames(target);
        for (var i = 0; i < props.length; i++) {
          var p = props[i];
          if (p === "window" || p === "document" || p === "top" || p === "parent" || p === "self") continue;
          try {
            var val = target[p];
            if (!val) continue;

            var isSocket = (typeof WebSocket !== "undefined" && val instanceof WebSocket) ||
              (typeof val === "object" && typeof val.send === "function" && typeof val.close === "function" && typeof val.readyState === "number");

            if (isSocket) {
              hookSocketInstance(val, "memory_scan:" + path + "." + p);
            } else if (typeof val === "object" && depth < 2 && !Array.isArray(val)) {
              walk(val, depth + 1, path + "." + p);
            }
          } catch (e) {}
        }
      } catch (e) {}
    }

    walk(globalObj, 0, "window");

    // Scan child iframes
    try {
      if (typeof document !== "undefined") {
        var iframes = document.querySelectorAll("iframe");
        for (var i = 0; i < iframes.length; i++) {
          try {
            if (iframes[i].contentWindow) {
              patchWSConstructor(iframes[i].contentWindow);
              walk(iframes[i].contentWindow, 0, "iframe[" + i + "]");
            }
          } catch (e) {}
        }
      }
    } catch (e) {}

    return discoveredSockets.length - countBefore;
  }

  // Run initial scan immediately
  scanForExistingSockets();

  // Periodic rescan to catch delayed sockets or reconnects
  if (!globalObj.__aviator_scan_interval) {
    globalObj.__aviator_scan_interval = setInterval(function () {
      scanForExistingSockets();
    }, 3000);
  }

  // =========================================================================
  // 6. PUBLIC API & TEST FUNCTIONS
  // =========================================================================
  function sendAviatorTestFrame(multiplier, roundId) {
    var m = multiplier !== undefined && multiplier !== null ? Number(multiplier) : 2.45;
    var r = roundId !== undefined && roundId !== null ? roundId : Date.now();
    var testPayload = JSON.stringify({
      type: "crash",
      round_id: r,
      multiplier: m,
      timestamp: new Date().toISOString()
    });

    console.log(
      "%c[Aviator ML Observer]%c 📤 Relaying test frame to Telemetry Bridge:",
      "color: #f59e0b; font-weight: bold;",
      "color: inherit;",
      { multiplier: m, roundId: r }
    );

    var delivered = postToBridge(testPayload);
    if (!delivered) {
      console.warn(
        "%c[Aviator ML Observer]%c ⚠️ Bridge companion window not open. Opening bridge...",
        "color: #f59e0b; font-weight: bold;",
        "color: inherit;"
      );
      openBridgeWindow();
    } else {
      console.log(
        "%c[Aviator ML Observer]%c ✅ Test frame dispatched to Bridge successfully!",
        "color: #10b981; font-weight: bold;",
        "color: inherit;"
      );
    }
    return {
      status: delivered ? "delivered_to_bridge" : "queued_awaiting_bridge",
      multiplier: m,
      roundId: r,
      bridgeConnected: isBridgeConnected || Boolean(findExistingBridge())
    };
  }

  function getAviatorDiagnostics() {
    return {
      observerInstalled: true,
      host: hostName,
      context: isIframe ? "iframe" : "top",
      bridgeConnected: isBridgeConnected || Boolean(findExistingBridge()),
      webSocketsDetectedCount: discoveredSockets.length,
      webSocketsDetected: discoveredSockets,
      rawFramesCapturedCount: rawFramesCapturedCount,
      framesSentToBridgeCount: framesSentCount,
      lastRawFrameSnippet: lastRawFrameSnippet
    };
  }

  function connectAviatorBridge() {
    return openBridgeWindow();
  }

  function getAviatorBridgeStatus() {
    var currentBridge = findExistingBridge();
    return {
      installed: true,
      bridgeConnected: Boolean(currentBridge && !currentBridge.closed),
      pendingFrames: pendingQueue.length,
      framesSent: framesSentCount,
      host: hostName,
      context: isIframe ? "iframe" : "top"
    };
  }

  // Expose globally on all scopes
  globalObj.sendAviatorTestFrame = sendAviatorTestFrame;
  globalObj.getAviatorDiagnostics = getAviatorDiagnostics;
  globalObj.scanAviatorWebSockets = scanForExistingSockets;
  globalObj.connectAviatorBridge = connectAviatorBridge;
  globalObj.getAviatorBridgeStatus = getAviatorBridgeStatus;

  if (typeof window !== "undefined") {
    window.sendAviatorTestFrame = sendAviatorTestFrame;
    window.getAviatorDiagnostics = getAviatorDiagnostics;
    window.scanAviatorWebSockets = scanForExistingSockets;
    window.connectAviatorBridge = connectAviatorBridge;
    window.getAviatorBridgeStatus = getAviatorBridgeStatus;
  }
  if (typeof globalThis !== "undefined") {
    globalThis.sendAviatorTestFrame = sendAviatorTestFrame;
    globalThis.getAviatorDiagnostics = getAviatorDiagnostics;
    globalThis.scanAviatorWebSockets = scanForExistingSockets;
    globalThis.connectAviatorBridge = connectAviatorBridge;
    globalThis.getAviatorBridgeStatus = getAviatorBridgeStatus;
  }
  if (typeof self !== "undefined") {
    self.sendAviatorTestFrame = sendAviatorTestFrame;
    self.getAviatorDiagnostics = getAviatorDiagnostics;
    self.scanAviatorWebSockets = scanForExistingSockets;
    self.connectAviatorBridge = connectAviatorBridge;
    self.getAviatorBridgeStatus = getAviatorBridgeStatus;
  }

  // =========================================================================
  // 7. INITIALIZATION & HEARTBEAT
  // =========================================================================
  openBridgeWindow();

  // Send installation event to bridge with initial diagnostics
  postToBridge({
    type: "observer_installed",
    installed: true,
    via_bridge: true,
    context: isIframe ? "iframe" : "top",
    host: hostName,
    sockets_detected: discoveredSockets.length,
    socket_urls: discoveredSockets.map(function (s) { return s.url; }),
    url: typeof window !== "undefined" && window.location ? window.location.href : "",
    timestamp: new Date().toISOString()
  });

  // Periodic heartbeat reporting live socket diagnostics
  if (!globalObj.__aviator_ping_interval) {
    globalObj.__aviator_ping_interval = setInterval(function () {
      postToBridge({
        type: "observer_ping",
        installed: true,
        via_bridge: true,
        sockets_detected: discoveredSockets.length,
        socket_urls: discoveredSockets.map(function (s) { return s.url; }),
        raw_frames_captured: rawFramesCapturedCount,
        timestamp: new Date().toISOString()
      });
      flushPendingQueue();
    }, 4000);
  }

  // Console banner
  console.log(
    "%c[Aviator ML Observer] ✅ TELEMETRY BRIDGE OBSERVER ACTIVE!%c\n" +
    "• Host Context: " + hostName + " (" + (isIframe ? "iframe" : "top") + ")\n" +
    "• Sockets Hooked: " + discoveredSockets.length + " (" + (discoveredSockets.length > 0 ? discoveredSockets.map(function(s){return s.url;}).join(", ") : "hooked via prototype.send") + ")\n" +
    "• window.sendAviatorTestFrame: function\n" +
    "• window.getAviatorDiagnostics(): inspect live sockets & frame counters\n" +
    "👉 Run %cgetAviatorDiagnostics()%c in console to check socket status!",
    "color: #10b981; font-weight: bold; font-size: 13px;",
    "color: inherit; font-size: 12px;",
    "color: #38bdf8; font-family: monospace; font-weight: bold;",
    "color: inherit;"
  );

  return "Aviator Observer Active";
})(typeof window !== "undefined" ? window : this);
`;



document.addEventListener("DOMContentLoaded", () => {
  initThemeToggle();
  initModeNavigation();
  initLiveControls();
  initObserverMonitorHandlers();
  initSourceModal();
  initCsvFallbackHandlers();

  // Load initial live state immediately
  fetchLiveState();

  // Establish live WebSocket connection to receive real-time round broadcasts
  connectLiveWebSocket();
});

// ==============================================================================
// THEME TOGGLE (Modern Web Guidance Compliant)
// ==============================================================================
function initThemeToggle() {
  const meta = document.querySelector('meta[name="color-scheme"]');
  const toggleBtn = document.getElementById("themeToggleBtn");
  if (!meta || !toggleBtn) return;

  function getEffectiveTheme() {
    if (meta.content === "dark" || meta.content === "light") {
      return meta.content;
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  toggleBtn.addEventListener("click", () => {
    const current = getEffectiveTheme();
    const nextTheme = current === "dark" ? "light" : "dark";
    meta.content = nextTheme;
    try {
      localStorage.setItem("color-scheme", nextTheme);
    } catch (e) {}

    // Redraw charts with active theme palette
    if (lastLiveState) {
      renderLiveVisualizations(lastLiveState);
    }
  });

  window.addEventListener("storage", (e) => {
    if (e.key === "color-scheme" && e.newValue) {
      meta.content = e.newValue;
    }
  });
}

// ==============================================================================
// MODE SWITCH: LIVE (PRIMARY) vs CSV (FALLBACK)
// ==============================================================================
function initModeNavigation() {
  const tabLive = document.getElementById("tabLiveBtn");
  const tabCsv = document.getElementById("tabCsvBtn");
  const liveView = document.getElementById("liveView");
  const csvView = document.getElementById("csvView");

  if (!tabLive || !tabCsv || !liveView || !csvView) return;

  tabLive.addEventListener("click", () => {
    currentMode = "live";
    tabLive.classList.add("active");
    tabLive.setAttribute("aria-selected", "true");
    tabCsv.classList.remove("active");
    tabCsv.setAttribute("aria-selected", "false");

    liveView.style.display = "block";
    csvView.style.display = "none";
  });

  tabCsv.addEventListener("click", () => {
    currentMode = "csv";
    tabCsv.classList.add("active");
    tabCsv.setAttribute("aria-selected", "true");
    tabLive.classList.remove("active");
    tabLive.setAttribute("aria-selected", "false");

    csvView.style.display = "block";
    liveView.style.display = "none";
  });
}

// ==============================================================================
// LIVE CONNECTION & CONTROLS
// ==============================================================================
function initLiveControls() {
  const connectBtn = document.getElementById("liveConnectBtn");
  if (!connectBtn) return;

  connectBtn.addEventListener("click", async () => {
    const isCurrentlyConnected = connectBtn.classList.contains("connected");
    if (isCurrentlyConnected) {
      // Disconnect
      await disconnectLiveSource();
    } else {
      // Connect to selected source (browser_observer by default if selected or active)
      const selectedSource = document.querySelector('input[name="sourceChoice"]:checked')?.value || "browser_observer";
      await connectLiveSource(selectedSource);
    }
  });
}

function initObserverMonitorHandlers() {
  const copyBtn = document.getElementById("copySnippetBtn");
  const copyDirectBtn = document.getElementById("copySnippetDirectBtn");
  const copyModalBtn = document.getElementById("copyModalSnippetBtn");
  const toggleGuideBtn = document.getElementById("toggleGuideBtn");
  const instructionsCard = document.getElementById("observerInstructionsCard");
  const toggleGuideText = document.getElementById("toggleGuideText");
  const textarea = document.getElementById("observerSnippetTextarea");
  let activeSnippetCode = AVIATOR_OBSERVER_SNIPPET;

  if (textarea) {
    textarea.value = AVIATOR_OBSERVER_SNIPPET;
  }

  // Fetch freshest observer script served by backend
  fetch("/api/live/observer.js")
    .then(res => res.ok ? res.text() : "")
    .then(code => {
      if (code && code.trim().length > 100) {
        activeSnippetCode = code;
        if (textarea) textarea.value = code;
      }
    })
    .catch(() => {});

  function handleCopy(btn) {
    if (!btn) return;
    const originalHtml = btn.innerHTML;

    function markSuccess() {
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M13.854 3.646a.5.5 0 0 1 0 .708l-7 7a.5.5 0 0 1-.708 0l-3.5-3.5a.5.5 0 1 1 .708-.708L6.5 10.293l6.646-6.647a.5.5 0 0 1 .708 0z"/></svg> Copied to Clipboard!`;
      setTimeout(() => {
        btn.innerHTML = originalHtml;
      }, 2500);
    }

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(activeSnippetCode || AVIATOR_OBSERVER_SNIPPET)
        .then(markSuccess)
        .catch(() => {
          fallbackCopyText(btn, originalHtml);
        });
    } else {
      fallbackCopyText(btn, originalHtml);
    }
  }

  function fallbackCopyText(btn, originalHtml) {
    if (textarea) {
      textarea.select();
      textarea.setSelectionRange(0, 99999);
      try {
        document.execCommand("copy");
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M13.854 3.646a.5.5 0 0 1 0 .708l-7 7a.5.5 0 0 1-.708 0l-3.5-3.5a.5.5 0 1 1 .708-.708L6.5 10.293l6.646-6.647a.5.5 0 0 1 .708 0z"/></svg> Copied!`;
        setTimeout(() => { btn.innerHTML = originalHtml; }, 2500);
        return;
      } catch (e) {}
    }
    prompt("Copy Aviator Observer snippet (Ctrl+C, Enter):", AVIATOR_OBSERVER_SNIPPET);
  }

  const launchBridgeBtn = document.getElementById("launchBridgeBtn");
  if (launchBridgeBtn) {
    launchBridgeBtn.addEventListener("click", () => {
      window.open("/bridge.html", "AviatorTelemetryBridge", "width=520,height=520");
    });
  }

  if (copyBtn) copyBtn.addEventListener("click", () => handleCopy(copyBtn));
  if (copyDirectBtn) copyDirectBtn.addEventListener("click", () => handleCopy(copyDirectBtn));
  if (copyModalBtn) copyModalBtn.addEventListener("click", () => handleCopy(copyModalBtn));

  if (toggleGuideBtn && instructionsCard) {
    toggleGuideBtn.addEventListener("click", () => {
      const isHidden = instructionsCard.style.display === "none";
      instructionsCard.style.display = isHidden ? "block" : "none";
      toggleGuideBtn.setAttribute("aria-expanded", isHidden ? "true" : "false");
      if (toggleGuideText) {
        toggleGuideText.textContent = isHidden ? "Hide Setup Instructions" : "Show Setup Instructions";
      }
    });
  }
}

async function connectLiveSource(sourceType, extra = {}) {
  try {
    const res = await fetch("/api/live/connect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source: sourceType, ...extra }),
    });
    const state = await res.json();
    renderLiveDashboard(state);
  } catch (err) {
    console.error("Failed to connect live source:", err);
  }
}

async function disconnectLiveSource() {
  try {
    const res = await fetch("/api/live/disconnect", { method: "POST" });
    const state = await res.json();
    renderLiveDashboard(state);
  } catch (err) {
    console.error("Failed to disconnect live source:", err);
  }
}

async function fetchLiveState() {
  try {
    const res = await fetch("/api/live/state");
    const state = await res.json();
    renderLiveDashboard(state);
  } catch (err) {
    console.error("Failed to fetch live state:", err);
  }
}

// WebSocket client for real-time live telemetry broadcasts
function connectLiveWebSocket() {
  if (liveWs && (liveWs.readyState === WebSocket.OPEN || liveWs.readyState === WebSocket.CONNECTING)) {
    return;
  }

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/api/live/ws`;

  try {
    liveWs = new WebSocket(wsUrl);

    liveWs.onopen = () => {
      console.log("[Aviator Live WS] Telemetry socket connected.");
      if (liveReconnectTimer) {
        clearInterval(liveReconnectTimer);
        liveReconnectTimer = null;
      }
    };

    liveWs.onmessage = (event) => {
      try {
        const state = JSON.parse(event.data);
        renderLiveDashboard(state);
      } catch (e) {
        console.error("Error parsing live WS payload:", e);
      }
    };

    liveWs.onclose = () => {
      console.log("[Aviator Live WS] Socket closed. Reconnecting in 3s...");
      scheduleReconnect();
    };

    liveWs.onerror = () => {
      liveWs.close();
    };
  } catch (err) {
    console.warn("WebSocket init error:", err);
    scheduleReconnect();
  }
}

function scheduleReconnect() {
  if (!liveReconnectTimer) {
    liveReconnectTimer = setTimeout(() => {
      liveReconnectTimer = null;
      connectLiveWebSocket();
    }, 3000);
  }
}

// ==============================================================================
// RENDER LIVE DASHBOARD
// ==============================================================================
function renderLiveDashboard(state) {
  if (!state) return;
  lastLiveState = state;

  const isConn = state.status === "connected";

  // 1. Live Connect Button
  const connectBtn = document.getElementById("liveConnectBtn");
  const connectText = document.getElementById("liveConnectText");
  if (connectBtn && connectText) {
    if (isConn) {
      connectBtn.classList.add("connected");
      connectBtn.setAttribute("aria-pressed", "true");
      connectText.textContent = "Disconnect Stream";
    } else {
      connectBtn.classList.remove("connected");
      connectBtn.setAttribute("aria-pressed", "false");
      connectText.textContent = "Connect Live Data";
    }
  }

  // 2. Stage Badge
  const stageBadge = document.getElementById("liveStageBadge");
  const stageText = document.getElementById("liveStageText");
  if (stageBadge && stageText) {
    const stage = state.current_stage || "idle";
    stageBadge.className = `stage-badge stage-${stage}`;
    stageText.textContent = stage.toUpperCase();
  }

  // 3. Live Summary Bar & Observer Diagnostic Check (Requirement 3, 4, 6)
  const connDot = document.getElementById("liveConnDot");
  const connStatus = document.getElementById("liveConnStatus");
  const obs = state.observer_telemetry || {};
  const obsInstalled = Boolean(obs.installed);
  const obsBridgeConnected = Boolean(obs.bridge_connected);
  const framesReceived = Number(obs.frames_received || 0);
  const parsedEvents = Number(obs.parsed_events || 0);
  const lastType = obs.last_event_type || "—";
  const lastRoundId = obs.last_round_id ? `#${obs.last_round_id}` : "—";
  const lastMult = obs.last_multiplier !== null && obs.last_multiplier !== undefined ? `${Number(obs.last_multiplier).toFixed(2)}x` : "—";
  const diagReason = obs.diagnostic_reason || "Observer snippet not detected.";

  if (connDot && connStatus) {
    if (state.source_type === "browser_observer") {
      if (framesReceived === 0) {
        connDot.className = "status-indicator-dot status-dot-warning";
        if (obsInstalled && obsBridgeConnected) {
          connStatus.textContent = "BRIDGE CONNECTED (Waiting for frames)";
        } else if (obsInstalled) {
          connStatus.textContent = "OBSERVER INSTALLED (Bridge Disconnected)";
        } else if (obsBridgeConnected) {
          connStatus.textContent = "BRIDGE READY (Observer Not Installed)";
        } else {
          connStatus.textContent = "BRIDGE DISCONNECTED (0 Frames)";
        }
      } else {
        connDot.className = "status-indicator-dot status-dot-online";
        connStatus.textContent = `LIVE (Stake Observer • ${framesReceived} frames)`;
      }
    } else if (isConn) {
      connDot.className = "status-indicator-dot status-dot-online";
      connStatus.textContent = `LIVE (${state.source_type || "stream"})`;
    } else {
      connDot.className = "status-indicator-dot status-dot-offline";
      connStatus.textContent = "Disconnected";
    }
  }

  // Monitor Card Badges: Installed, Bridge, and Receiving (Requirement 3)
  const obsInstalledBadge = document.getElementById("obsInstalledBadge");
  const obsBridgeBadge = document.getElementById("obsBridgeBadge");
  const obsReceivingBadge = document.getElementById("obsReceivingBadge");

  if (obsInstalledBadge) {
    if (obsInstalled) {
      obsInstalledBadge.className = "badge badge-success";
      obsInstalledBadge.textContent = `Observer: Installed (${obs.host || "aviator-next"})`;
    } else {
      obsInstalledBadge.className = "badge badge-warning";
      obsInstalledBadge.textContent = "Observer: Not Installed";
    }
  }

  if (obsBridgeBadge) {
    if (obsBridgeConnected) {
      obsBridgeBadge.className = "badge badge-success";
      obsBridgeBadge.textContent = "Bridge: Connected";
    } else {
      obsBridgeBadge.className = "badge badge-warning";
      obsBridgeBadge.textContent = "Bridge: Disconnected";
    }
  }

  if (obsReceivingBadge) {
    if (framesReceived > 0) {
      obsReceivingBadge.className = "badge badge-success";
      obsReceivingBadge.textContent = `Frames: Receiving (${framesReceived} frames)`;
    } else {
      obsReceivingBadge.className = "badge badge-warning";
      obsReceivingBadge.textContent = "Frames: 0 Received";
    }
  }

  // Observer Telemetry Grid (Requirement 1 & 5)
  const obsMetricInstalled = document.getElementById("obsMetricInstalled");
  const obsMetricInstalledSub = document.getElementById("obsMetricInstalledSub");
  const obsMetricBridge = document.getElementById("obsMetricBridge");
  const obsMetricBridgeSub = document.getElementById("obsMetricBridgeSub");
  const obsMetricSockets = document.getElementById("obsMetricSockets");
  const obsMetricSocketsSub = document.getElementById("obsMetricSocketsSub");
  const obsMetricGameFrames = document.getElementById("obsMetricGameFrames");
  const obsMetricFrames = document.getElementById("obsMetricFrames");
  const obsMetricFramesSub = document.getElementById("obsMetricFramesSub");
  const obsMetricParserAttempts = document.getElementById("obsMetricParserAttempts");
  const obsMetricEvents = document.getElementById("obsMetricEvents");
  const obsMetricLastType = document.getElementById("obsMetricLastType");
  const obsMetricLastParsedSub = document.getElementById("obsMetricLastParsedSub");
  const obsMetricLastMultiplier = document.getElementById("obsMetricLastMultiplier");

  const socketsCount = Number(obs.detected_sockets_count || (obs.detected_socket_urls ? obs.detected_socket_urls.length : 0));
  const socketUrls = obs.detected_socket_urls || [];
  const rawGameFrames = Number(obs.raw_game_frames_captured || 0);
  const parserAttempts = Number(obs.parser_attempts || 0);

  if (obsMetricInstalled) {
    obsMetricInstalled.textContent = obsInstalled ? "YES" : "NO";
    obsMetricInstalled.style.color = obsInstalled ? "var(--color-success)" : "var(--color-warning)";
  }
  if (obsMetricInstalledSub) {
    obsMetricInstalledSub.textContent = obsInstalled ? (obs.host || "Aviator iframe") : "Not detected";
  }

  if (obsMetricBridge) {
    obsMetricBridge.textContent = obsBridgeConnected ? "CONNECTED" : "DISCONNECTED";
    obsMetricBridge.style.color = obsBridgeConnected ? "var(--color-success)" : "var(--color-warning)";
  }
  if (obsMetricBridgeSub) {
    obsMetricBridgeSub.textContent = obsBridgeConnected ? "Local gateway active" : "Launch bridge window";
  }

  if (obsMetricSockets) {
    obsMetricSockets.textContent = socketsCount;
    obsMetricSockets.style.color = socketsCount > 0 ? "var(--accent-primary)" : "var(--text-muted)";
  }
  if (obsMetricSocketsSub) {
    if (socketsCount > 0) {
      obsMetricSocketsSub.textContent = socketUrls.length > 0 ? socketUrls[0].replace(/^wss?:\/\//, "").slice(0, 24) : `${socketsCount} active`;
    } else {
      obsMetricSocketsSub.textContent = obsInstalled ? "Scanning sockets..." : "No active sockets";
    }
  }

  if (obsMetricGameFrames) {
    obsMetricGameFrames.textContent = rawGameFrames;
    obsMetricGameFrames.style.color = rawGameFrames > 0 ? "var(--color-success)" : "var(--text-muted)";
  }

  if (obsMetricFrames) obsMetricFrames.textContent = framesReceived;
  if (obsMetricFramesSub) {
    obsMetricFramesSub.textContent = framesReceived > 0 ? "Ingested via bridge" : "0 frames ingested";
  }

  if (obsMetricParserAttempts) obsMetricParserAttempts.textContent = parserAttempts;
  if (obsMetricEvents) obsMetricEvents.textContent = parsedEvents;
  if (obsMetricLastType) obsMetricLastType.textContent = lastType;
  if (obsMetricLastParsedSub) {
    if (obs.last_parsed_event && obs.last_parsed_event.event_type) {
      obsMetricLastParsedSub.textContent = `Parsed: ${obs.last_parsed_event.event_type}`;
    } else if (obs.last_raw_frame_snippet) {
      obsMetricLastParsedSub.textContent = "Raw snippet logged";
    } else {
      obsMetricLastParsedSub.textContent = "Event classification";
    }
  }
  if (obsMetricLastMultiplier) obsMetricLastMultiplier.textContent = lastMult;

  // 3-Stage Stepper: Stage 1 Observer Installed -> Stage 2 Bridge Connected -> Stage 3 Frames Received (Requirement 3)
  const stepInstalled = document.getElementById("stepInstalled") || document.getElementById("stepNotDetected");
  const stepBridge = document.getElementById("stepBridge") || document.getElementById("stepConnected");
  const stepFramesReceived = document.getElementById("stepFramesReceived");
  const connector1 = document.getElementById("connector1");
  const connector2 = document.getElementById("connector2");

  if (stepInstalled && stepBridge && stepFramesReceived && connector1 && connector2) {
    stepInstalled.classList.remove("step-active", "step-completed");
    stepBridge.classList.remove("step-active", "step-completed");
    stepFramesReceived.classList.remove("step-active", "step-completed");
    connector1.classList.remove("connector-active");
    connector2.classList.remove("connector-active");

    if (framesReceived > 0) {
      stepInstalled.classList.add("step-completed");
      stepBridge.classList.add("step-completed");
      stepFramesReceived.classList.add("step-active");
      connector1.classList.add("connector-active");
      connector2.classList.add("connector-active");
    } else if (obsBridgeConnected && obsInstalled) {
      stepInstalled.classList.add("step-completed");
      stepBridge.classList.add("step-active");
      connector1.classList.add("connector-active");
    } else if (obsInstalled) {
      stepInstalled.classList.add("step-active");
    } else if (obsBridgeConnected) {
      stepBridge.classList.add("step-active");
    } else {
      stepInstalled.classList.add("step-active");
    }
  }

  // Diagnostic Reason Banner (Requirement 4 & 6)
  const obsDiagnosticHeading = document.getElementById("obsDiagnosticHeading");
  const obsDiagnosticReason = document.getElementById("obsDiagnosticReason");
  const obsDiagnosticBanner = document.getElementById("obsDiagnosticBanner");

  if (obsDiagnosticHeading && obsDiagnosticReason && obsDiagnosticBanner) {
    obsDiagnosticReason.textContent = diagReason;
    if (parsedEvents > 0) {
      obsDiagnosticHeading.textContent = "Live Telemetry Operational: Processing Real Aviator Game Frames";
      obsDiagnosticBanner.className = "obs-diagnostic-alert alert-success";
    } else if (rawGameFrames > 0) {
      obsDiagnosticHeading.textContent = "Diagnostic: Raw Game Frames Ingested (Awaiting parser lifecycle match)";
      obsDiagnosticBanner.className = "obs-diagnostic-alert alert-info";
    } else if (framesReceived > 0) {
      obsDiagnosticHeading.textContent = "Diagnostic: Bridge Heartbeat Received (Waiting for Aviator game frames)";
      obsDiagnosticBanner.className = "obs-diagnostic-alert alert-info";
    } else if (obsInstalled && obsBridgeConnected) {
      obsDiagnosticHeading.textContent = "Diagnostic: Observer Installed & Bridge Connected (Waiting for round events)";
      obsDiagnosticBanner.className = "obs-diagnostic-alert alert-info";
    } else if (!obsBridgeConnected && obsInstalled) {
      obsDiagnosticHeading.textContent = "Diagnostic: Telemetry Bridge Disconnected (Launch Bridge Window)";
      obsDiagnosticBanner.className = "obs-diagnostic-alert alert-warning";
    } else if (obsBridgeConnected && !obsInstalled) {
      obsDiagnosticHeading.textContent = "Diagnostic: Bridge Connected (Waiting for Observer in DevTools)";
      obsDiagnosticBanner.className = "obs-diagnostic-alert alert-warning";
    } else {
      obsDiagnosticHeading.textContent = "Diagnostic: Telemetry Bridge & Observer Not Connected";
      obsDiagnosticBanner.className = "obs-diagnostic-alert alert-warning";
    }
  }

  const elRoundId = document.getElementById("liveRoundId");
  if (elRoundId) elRoundId.textContent = state.current_round_id ? `#${state.current_round_id}` : "—";

  const elLatestMult = document.getElementById("liveLatestMult");
  if (elLatestMult) elLatestMult.textContent = state.latest_multiplier !== null && state.latest_multiplier !== undefined ? `${state.latest_multiplier.toFixed(2)}x` : "—";

  const metrics = state.running_metrics || {};
  const elRoundsCount = document.getElementById("liveRoundsCount");
  if (elRoundsCount) elRoundsCount.textContent = `${metrics.total_rounds || 0}`;

  const elBrier = document.getElementById("liveBrierVal");
  if (elBrier) elBrier.textContent = metrics.brier_score_2x !== undefined ? metrics.brier_score_2x.toFixed(4) : "0.2500";

  const elLogLoss = document.getElementById("liveLogLossVal");
  if (elLogLoss) elLogLoss.textContent = metrics.log_loss_2x !== undefined ? metrics.log_loss_2x.toFixed(4) : "0.6931";

  const elHitRate = document.getElementById("liveHitRateVal");
  if (elHitRate) {
    const hitPct = ((metrics.actual_hit_rate_2x || 0) * 100).toFixed(1);
    elHitRate.innerHTML = `${hitPct}% <span class="text-dim text-xs">(exp. 48.5%)</span>`;
  }

  // 4. Scrolling History Strip
  renderHistoryStrip(state.history_strip || []);

  // 5. Main Next-Round Prediction Hero Card
  renderNextRoundHero(state.active_prediction);

  // 6. Prediction vs Actual Tracker Table
  renderEvaluationTable(state.evaluation_history || []);

  // 7. Prediction Overview: 6 ML Models
  if (state.active_prediction && state.active_prediction.model_predictions) {
    renderModelsGrid(state.active_prediction.model_predictions);
  }

  // 8. Model Comparison Benchmark Table
  renderBenchmarkTable(state.benchmark_comparison || []);

  // 9. Visualizations
  renderLiveVisualizations(state);
}

// Scrolling History Strip Renderer
function renderHistoryStrip(stripRecords) {
  const container = document.getElementById("historyStrip");
  if (!container) return;

  if (stripRecords.length === 0) {
    container.innerHTML = `<span class="text-dim text-xs">Awaiting round telemetry...</span>`;
    return;
  }

  container.innerHTML = "";
  // Show most recent rounds
  stripRecords.forEach((r) => {
    const chip = document.createElement("span");
    const m = r.multiplier;
    let chipType = "chip-low";
    if (m >= 10.0) chipType = "chip-high";
    else if (m >= 2.0) chipType = "chip-medium";

    chip.className = `strip-chip ${chipType}`;
    chip.textContent = `${m.toFixed(2)}x`;
    chip.title = `Round #${r.round_id} — ${m.toFixed(2)}x`;
    container.appendChild(chip);
  });

  // Scroll to end automatically
  container.scrollLeft = container.scrollWidth;
}

// Next Round Hero Prediction Card
function renderNextRoundHero(pred) {
  if (!pred) return;

  const elNextId = document.getElementById("nextRoundId");
  if (elNextId) elNextId.textContent = pred.round_id || "—";

  const elCal = document.getElementById("calibrationStatusText");
  if (elCal) elCal.textContent = pred.calibration_status || "Well-Calibrated";

  const grid = document.getElementById("mainThresholdsGrid");
  if (!grid) return;
  grid.innerHTML = "";

  const thresholds = [
    { key: "1.5x", label: "P(≥ 1.5x)", fair: "64.7%" },
    { key: "2.0x", label: "P(≥ 2.0x)", fair: "48.5%" },
    { key: "3.0x", label: "P(≥ 3.0x)", fair: "32.3%" },
    { key: "5.0x", label: "P(≥ 5.0x)", fair: "19.4%" },
    { key: "10.0x", label: "P(≥ 10.0x)", fair: "9.7%" },
  ];

  thresholds.forEach((t) => {
    const ensProb = pred.ensemble_probability ? pred.ensemble_probability[t.key] : null;
    const meanProb = pred.consensus_mean ? pred.consensus_mean[t.key] : 0.5;
    const displayProb = ensProb !== null && ensProb !== undefined ? ensProb : meanProb;
    const pct = (displayProb * 100).toFixed(1);

    const card = document.createElement("div");
    card.className = "threshold-card";
    card.innerHTML = `
      <span class="threshold-target">${t.label}</span>
      <span class="threshold-prob">${pct}%</span>
      <div class="progress-bar-bg" aria-hidden="true">
        <div class="progress-bar-fill" style="width: ${pct}%"></div>
      </div>
      <span class="threshold-benchmark font-mono">Fair RTP: ${t.fair}</span>
    `;
    grid.appendChild(card);
  });

  // Consensus Details
  if (pred.ensemble_probability) {
    const p2x = pred.ensemble_probability["2.0x"];
    const elEns = document.getElementById("consensusEnsembleVal");
    if (elEns) elEns.textContent = p2x !== undefined ? `${(p2x * 100).toFixed(1)}%` : "N/A";
  }
  if (pred.consensus_mean) {
    const m2x = pred.consensus_mean["2.0x"];
    const elMean = document.getElementById("consensusMeanVal");
    if (elMean) elMean.textContent = m2x !== undefined ? `${(m2x * 100).toFixed(1)}%` : "N/A";
  }
  if (pred.consensus_min && pred.consensus_max) {
    const min2x = (pred.consensus_min["2.0x"] * 100).toFixed(1);
    const max2x = (pred.consensus_max["2.0x"] * 100).toFixed(1);
    const elSpread = document.getElementById("consensusSpreadVal");
    if (elSpread) elSpread.textContent = `${min2x}% — ${max2x}%`;
  }
}

// Prediction vs Actual Tracker Table
function renderEvaluationTable(evalList) {
  const tbody = document.getElementById("liveEvalTableBody");
  if (!tbody) return;

  if (evalList.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" class="text-muted text-center py-4">Awaiting completed rounds in this session...</td></tr>`;
    return;
  }

  tbody.innerHTML = "";
  // Show in reverse order (newest on top)
  const reversed = [...evalList].reverse();

  reversed.forEach((row) => {
    const tr = document.createElement("tr");
    const isHit = row.actual_multiplier >= 2.0;
    const badgeClass = isHit ? "badge-hit" : "badge-miss";
    const tsStr = row.timestamp ? row.timestamp.slice(11, 19) : "—";

    tr.innerHTML = `
      <td class="font-mono"><strong>#${row.round_id}</strong></td>
      <td class="text-right font-mono">${(row.predicted_prob_2x * 100).toFixed(1)}%</td>
      <td class="text-right font-mono"><span class="multiplier-pill ${isHit ? 'multiplier-high' : 'multiplier-low'}">${row.actual_multiplier.toFixed(2)}x</span></td>
      <td><span class="badge ${badgeClass}">${isHit ? 'HIT (≥2.0x)' : 'CRASH (<2.0x)'}</span></td>
      <td class="text-right font-mono text-muted">${row.brier_error.toFixed(4)}</td>
      <td class="text-muted text-xs font-mono">${tsStr}</td>
    `;
    tbody.appendChild(tr);
  });
}

// The 6 ML Models Overview Grid
function renderModelsGrid(predictions) {
  const container = document.getElementById("liveModelsGrid");
  if (!container) return;
  container.innerHTML = "";

  const modelSpecs = [
    { key: "Logistic Regression", tag: "L2 Regularized C=1.0" },
    { key: "Random Forest", tag: "200 Trees • Max Depth 6" },
    { key: "XGBoost", tag: "Gradient Boosted Trees" },
    { key: "MLP", tag: "PyTorch [64, 32] Hidden" },
    { key: "LSTM", tag: "PyTorch Recurrent Window=10" },
    { key: "GRU", tag: "PyTorch Gated Recurrent" },
  ];

  const thresholds = ["1.5x", "2.0x", "3.0x", "5.0x", "10.0x"];

  modelSpecs.forEach((m) => {
    const card = document.createElement("div");
    card.className = "model-card";
    const modelPreds = predictions[m.key] || {};

    let listHtml = "";
    thresholds.forEach((th) => {
      const prob = modelPreds[th] !== undefined ? modelPreds[th] : 0.0;
      const pct = (prob * 100).toFixed(1);
      listHtml += `
        <div class="model-threshold-row">
          <span class="model-threshold-label">P(≥${th})</span>
          <div class="model-threshold-bar-wrap">
            <div class="progress-bar-bg">
              <div class="progress-bar-fill" style="width: ${pct}%"></div>
            </div>
          </div>
          <span class="model-threshold-val">${pct}%</span>
        </div>
      `;
    });

    card.innerHTML = `
      <div class="model-card-header">
        <h3 class="model-card-name">${m.key}</h3>
        <span class="model-card-type">${m.tag}</span>
      </div>
      <div class="model-thresholds-list">
        ${listHtml}
      </div>
    `;

    container.appendChild(card);
  });
}

// Model Comparison Benchmark Table
function renderBenchmarkTable(benchmarkData) {
  const tbody = document.getElementById("liveComparisonTableBody");
  if (!tbody) return;
  tbody.innerHTML = "";

  if (!benchmarkData || benchmarkData.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" class="text-muted">No benchmark records available.</td></tr>`;
    return;
  }

  benchmarkData.forEach((row) => {
    const tr = document.createElement("tr");
    if (row.is_strongest) tr.className = "row-strongest";

    const badgeHtml = row.is_strongest
      ? `<span class="badge badge-strongest">Highest AUC</span>`
      : row.category === "Baseline"
      ? `<span class="badge" style="background: var(--bg-card-subtle); color: var(--text-muted); border: 1px solid var(--border-subtle);">Baseline</span>`
      : `<span class="badge" style="background: var(--bg-card-subtle); color: var(--accent-primary); border: 1px solid var(--border-subtle);">Verified</span>`;

    tr.innerHTML = `
      <td><strong>${row.model}</strong></td>
      <td class="text-right font-mono">${row.roc_auc.toFixed(4)}</td>
      <td class="text-right font-mono">${row.brier_score.toFixed(4)}</td>
      <td class="text-right font-mono">${row.log_loss.toFixed(4)}</td>
      <td>${badgeHtml}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ==============================================================================
// VISUALIZATIONS (Native Responsive SVG Charts)
// ==============================================================================
function renderLiveVisualizations(state) {
  const history = state.history_strip || [];
  renderLiveHistoryChart(history);
  renderLiveThresholdsChart(state.active_prediction);
}

// Chart 1: Live Multiplier History
function renderLiveHistoryChart(history) {
  const container = document.getElementById("liveChartHistoryContainer");
  if (!container || history.length === 0) return;

  const width = container.clientWidth || 500;
  const height = container.clientHeight || 240;
  const padding = { top: 20, right: 20, bottom: 30, left: 45 };

  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const rawMax = Math.max(...history.map((d) => d.multiplier), 5.0);
  const maxM = Math.min(rawMax, 15.0);

  const points = history.map((d, i) => {
    const x = padding.left + (i / Math.max(1, history.length - 1)) * plotW;
    const val = Math.min(d.multiplier, maxM);
    const y = padding.top + plotH - (val / maxM) * plotH;
    return { x, y, round_id: d.round_id, mult: d.multiplier };
  });

  const polylinePoints = points.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const y2x = padding.top + plotH - (2.0 / maxM) * plotH;

  let svg = `
    <svg viewBox="0 0 ${width} ${height}" width="100%" height="100%" style="overflow: visible;">
      <line x1="${padding.left}" y1="${padding.top + plotH}" x2="${width - padding.right}" y2="${padding.top + plotH}" stroke="var(--border-subtle)" stroke-width="1"/>
      <line x1="${padding.left}" y1="${y2x}" x2="${width - padding.right}" y2="${y2x}" stroke="var(--color-warning)" stroke-width="1" stroke-dasharray="4 4" stroke-opacity="0.8"/>
      <text x="${width - padding.right}" y="${y2x - 4}" fill="var(--color-warning)" font-size="10" font-family="var(--font-mono)" text-anchor="end">2.0x Reference</text>

      <text x="${padding.left - 6}" y="${padding.top + plotH}" fill="var(--text-dim)" font-size="10" font-family="var(--font-mono)" text-anchor="end">0x</text>
      <text x="${padding.left - 6}" y="${padding.top + 10}" fill="var(--text-dim)" font-size="10" font-family="var(--font-mono)" text-anchor="end">${maxM.toFixed(0)}x</text>
      <text x="${padding.left}" y="${height - 8}" fill="var(--text-dim)" font-size="10" font-family="var(--font-mono)">-${history.length}</text>
      <text x="${width - padding.right}" y="${height - 8}" fill="var(--text-dim)" font-size="10" font-family="var(--font-mono)" text-anchor="end">Latest</text>

      <polyline points="${polylinePoints}" fill="none" stroke="var(--accent-primary)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  `;

  points.forEach((p) => {
    const color = p.mult >= 2.0 ? "var(--color-success)" : "var(--accent-primary)";
    svg += `
      <circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="3" fill="${color}" stroke="var(--bg-card)" stroke-width="1.5">
        <title>Round #${p.round_id}: ${p.mult.toFixed(2)}x</title>
      </circle>
    `;
  });

  svg += `</svg>`;
  container.innerHTML = svg;
}

// Chart 2: Probability by Threshold
function renderLiveThresholdsChart(pred) {
  const container = document.getElementById("liveChartThresholdsContainer");
  if (!container || !pred) return;

  const width = container.clientWidth || 500;
  const height = container.clientHeight || 240;
  const padding = { top: 20, right: 20, bottom: 35, left: 45 };

  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const thresholds = ["1.5x", "2.0x", "3.0x", "5.0x", "10.0x"];
  const fair = [0.647, 0.485, 0.323, 0.194, 0.097];
  const consensus = pred.consensus_mean || {};

  const colWidth = plotW / thresholds.length;

  let svg = `
    <svg viewBox="0 0 ${width} ${height}" width="100%" height="100%">
      <line x1="${padding.left}" y1="${padding.top}" x2="${width - padding.right}" y2="${padding.top}" stroke="var(--border-subtle)" stroke-width="1" stroke-dasharray="2 2"/>
      <text x="${padding.left - 6}" y="${padding.top + 4}" fill="var(--text-dim)" font-size="10" font-family="var(--font-mono)" text-anchor="end">100%</text>

      <line x1="${padding.left}" y1="${padding.top + plotH / 2}" x2="${width - padding.right}" y2="${padding.top + plotH / 2}" stroke="var(--border-subtle)" stroke-width="1" stroke-dasharray="2 2"/>
      <text x="${padding.left - 6}" y="${padding.top + plotH / 2 + 4}" fill="var(--text-dim)" font-size="10" font-family="var(--font-mono)" text-anchor="end">50%</text>

      <line x1="${padding.left}" y1="${padding.top + plotH}" x2="${width - padding.right}" y2="${padding.top + plotH}" stroke="var(--border-subtle)" stroke-width="1"/>
      <text x="${padding.left - 6}" y="${padding.top + plotH + 4}" fill="var(--text-dim)" font-size="10" font-family="var(--font-mono)" text-anchor="end">0%</text>
  `;

  thresholds.forEach((th, i) => {
    const xBase = padding.left + i * colWidth;
    const fairVal = fair[i];
    const meanVal = consensus[th] !== undefined ? consensus[th] : fairVal;

    const barW = Math.min(26, colWidth * 0.35);
    const xMean = xBase + colWidth / 2 - barW - 2;
    const xFair = xBase + colWidth / 2 + 2;

    const hMean = Math.min(plotH, Math.max(2, meanVal * plotH));
    const hFair = Math.min(plotH, Math.max(2, fairVal * plotH));

    const yMean = padding.top + plotH - hMean;
    const yFair = padding.top + plotH - hFair;

    svg += `
      <rect x="${xMean}" y="${yMean}" width="${barW}" height="${hMean}" rx="3" fill="var(--accent-primary)">
        <title>Mean Model Estimate P(≥${th}): ${(meanVal * 100).toFixed(1)}%</title>
      </rect>
      <rect x="${xFair}" y="${yFair}" width="${barW}" height="${hFair}" rx="3" fill="var(--border-subtle)">
        <title>Theoretical Fair Game P(≥${th}): ${(fairVal * 100).toFixed(1)}%</title>
      </rect>
      <text x="${xBase + colWidth / 2}" y="${height - 12}" fill="var(--text-muted)" font-size="11" font-weight="600" font-family="var(--font-mono)" text-anchor="middle">≥${th}</text>
    `;
  });

  svg += `
    <rect x="${width - 195}" y="2" width="10" height="10" rx="2" fill="var(--accent-primary)"/>
    <text x="${width - 180}" y="10" fill="var(--text-muted)" font-size="10" font-family="var(--font-sans)">Model Estimate</text>
    <rect x="${width - 95}" y="2" width="10" height="10" rx="2" fill="var(--border-subtle)"/>
    <text x="${width - 80}" y="10" fill="var(--text-muted)" font-size="10" font-family="var(--font-sans)">Fair Game</text>
  `;

  svg += `</svg>`;
  container.innerHTML = svg;
}

// ==============================================================================
// SOURCE CONFIGURATION MODAL
// ==============================================================================
function initSourceModal() {
  const modal = document.getElementById("sourceModal");
  const openBtn = document.getElementById("sourceConfigBtn");
  const closeBtn = document.getElementById("closeModalBtn");
  const applyBtn = document.getElementById("applySourceBtn");

  if (!modal || !openBtn || !closeBtn || !applyBtn) return;

  openBtn.addEventListener("click", () => modal.showModal());
  closeBtn.addEventListener("click", () => modal.close());
  modal.addEventListener("click", (e) => {
    if (e.target === modal) modal.close();
  });

  const radios = document.querySelectorAll('input[name="sourceChoice"]');
  const wsInputWrap = document.getElementById("wsUrlInputWrap");
  const snippetWrap = document.getElementById("observerSnippetWrap");

  radios.forEach((r) => {
    r.addEventListener("change", () => {
      if (r.value === "websocket") {
        wsInputWrap.style.display = "block";
        snippetWrap.style.display = "none";
      } else if (r.value === "browser_observer") {
        wsInputWrap.style.display = "none";
        snippetWrap.style.display = "block";
      } else {
        wsInputWrap.style.display = "none";
        snippetWrap.style.display = "none";
      }
    });
  });

  applyBtn.addEventListener("click", async () => {
    const selected = document.querySelector('input[name="sourceChoice"]:checked')?.value || "sample_stream";
    const wsUrl = document.getElementById("wsUrlInput")?.value || "";

    modal.close();
    await connectLiveSource(selected, { url: wsUrl });
  });
}

// ==============================================================================
// CSV FALLBACK HANDLERS
// ==============================================================================
function initCsvFallbackHandlers() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const btnReal = document.getElementById("btnSampleReal");
  const btnSynth = document.getElementById("btnSampleSynth");

  if (!dropzone || !fileInput) return;

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag-over");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleCsvUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleCsvUpload(e.target.files[0]);
    }
  });

  if (btnReal) {
    btnReal.addEventListener("click", () => fetchCsvSample("real"));
  }
  if (btnSynth) {
    btnSynth.addEventListener("click", () => fetchCsvSample("synthetic"));
  }
}

async function handleCsvUpload(file) {
  if (!file) return;
  hideCsvErrors();

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok || data.status === "error") {
      showCsvErrors(data.errors || [data.message || "Failed to process CSV file."]);
      return;
    }
    renderCsvResults(data);
  } catch (err) {
    showCsvErrors([`Network error: ${err.message}`]);
  }
}

async function fetchCsvSample(source) {
  hideCsvErrors();
  try {
    const res = await fetch(`/api/sample?source=${source}`);
    const data = await res.json();
    if (!res.ok || data.status === "error") {
      showCsvErrors(data.errors || [data.message || "Failed to load sample dataset."]);
      return;
    }
    renderCsvResults(data);
  } catch (err) {
    showCsvErrors([`Sample fetch error: ${err.message}`]);
  }
}

function renderCsvResults(data) {
  const summaryBox = document.getElementById("csvSummary");
  if (summaryBox && data.file_summary) {
    const sum = data.file_summary;
    document.getElementById("csvMetaFilename").textContent = sum.filename;
    document.getElementById("csvMetaRounds").textContent = `${sum.total_rounds.toLocaleString()} rounds`;
    document.getElementById("csvMetaTimeRange").textContent = `${sum.time_start.slice(0, 19).replace('T', ' ')} → ${sum.time_end.slice(0, 19).replace('T', ' ')}`;
    document.getElementById("csvMetaLatestMultiplier").textContent = `${sum.latest_multiplier.toFixed(2)}x`;
    summaryBox.style.display = "grid";
  }

  // Update live view with this uploaded data so user can see it in live view too!
  if (data.next_round_prediction) {
    renderNextRoundHero(data.next_round_prediction);
    if (data.model_predictions) renderModelsGrid(data.model_predictions);
  }
}

function showCsvErrors(errors) {
  const box = document.getElementById("csvErrorBox");
  const list = document.getElementById("csvErrorList");
  if (!box || !list) return;

  list.innerHTML = "";
  errors.forEach((err) => {
    const li = document.createElement("li");
    li.textContent = err;
    list.appendChild(li);
  });
  box.style.display = "flex";
}

function hideCsvErrors() {
  const box = document.getElementById("csvErrorBox");
  if (box) box.style.display = "none";
}
