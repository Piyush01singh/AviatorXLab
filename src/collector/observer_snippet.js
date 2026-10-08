/**
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
  function uint8ArrayToBase64(bytes) {
    if (!bytes) return null;
    try {
      var binary = "";
      var len = bytes.length;
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

    var b64 = bytes ? uint8ArrayToBase64(bytes) : null;

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
