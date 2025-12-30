// Chatbot Component - Reusable across all pages
// This creates a fixed chatbot button that expands into a chat interface

(function() {
  // Create chatbot styles and loadable config
  const defaultConfig = {
    endpoint: 'https://eden-chat.csirico9.workers.dev',
    initial_message: "Hello! I'm your Neurofoundry AI assistant. How can I help you today?",
    placeholder: 'Type your message...',
    window_width: 300,
    window_height: 360,
    accent1: '#ff7b6e',
    accent2: '#e0473c',
    bg: 'rgba(15, 17, 19, 0.98)',
    enable_tracing: true,
    typing_text: 'Typing...',
    error_message: "Sorry, I'm having trouble connecting. Please try again.",

    // Fuzzy matching defaults
    match_fuzzy: true,
    fuzzy_threshold: 0.8,    // similarity score 0..1
    fuzzy_whole_word: true,  // only compare against individual tokens when true

    // Assistant instruction defaults and response policy
    assistant_instructions: "You are the Neurofoundry assistant. Be concise (20-80 words), avoid long encyclopedia-like answers, and guide the user to sign up or explore site features when appropriate.",
    cta_text: "Want more detail? Sign up to get a full walkthrough:",
    cta_url: "/signup.html",
    max_response_length: 300,
    shorten_policy: true
  };
  let chatConfig = Object.assign({}, defaultConfig);

  // Simple Levenshtein distance + normalized similarity helper
  function levenshtein(a, b) {
    if (!a || !b) return (a || b) ? Math.max((a||'').length, (b||'').length) : 0;
    a = String(a); b = String(b);
    const la = a.length, lb = b.length;
    const dp = Array(la + 1).fill(null).map(() => Array(lb + 1).fill(0));
    for (let i = 0; i <= la; i++) dp[i][0] = i;
    for (let j = 0; j <= lb; j++) dp[0][j] = j;
    for (let i = 1; i <= la; i++) {
      for (let j = 1; j <= lb; j++) {
        const cost = a[i-1] === b[j-1] ? 0 : 1;
        dp[i][j] = Math.min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost);
      }
    }
    return dp[la][lb];
  }

  function similarity(a, b) {
    a = String(a || '').trim().toLowerCase();
    b = String(b || '').trim().toLowerCase();
    if (!a && !b) return 1;
    if (!a || !b) return 0;
    const dist = levenshtein(a, b);
    const maxLen = Math.max(a.length, b.length);
    return maxLen === 0 ? 1 : 1 - (dist / maxLen);
  }

  async function loadExternalConfig() {
    // Try YAML first, then JSON; merges into chatConfig
    try {
      const yamlRes = await fetch('/chatbot.config.yaml');
      if (yamlRes.ok) {
        const txt = await yamlRes.text();
        if (!window.jsyaml) {
          await new Promise((resolve, reject) => {
            const s = document.createElement('script');
            s.src = 'https://cdn.jsdelivr.net/npm/js-yaml@4.1.0/dist/js-yaml.min.js';
            s.onload = resolve; s.onerror = reject;
            document.head.appendChild(s);
          });
        }
        try {
          const cfg = (window.jsyaml && window.jsyaml.load) ? window.jsyaml.load(txt) : null;
          if (cfg && typeof cfg === 'object') { Object.assign(chatConfig, cfg); return chatConfig; }
        } catch(e) { console.warn('YAML parse failed', e); }
      }
    } catch(e) { /* ignore */ }

    try {
      const jsonRes = await fetch('/chatbot.config.json');
      if (jsonRes.ok) { const cfg = await jsonRes.json(); if (cfg) { Object.assign(chatConfig, cfg); return chatConfig; } }
    } catch(e) { /* ignore */ }

    return chatConfig;
  }

  const style = document.createElement('style');
  style.textContent = `
    :root {
      --accent-1: ${chatConfig.accent1};
      --accent-2: ${chatConfig.accent2};
      --bg: ${chatConfig.bg};
      --window-width: ${chatConfig.window_width}px;
      --window-height: ${chatConfig.window_height}px;
    }

    .chatbot-container {
      position: fixed;
      bottom: 24px;
      left: 24px;
      z-index: 9999;
      font-family: Inter, sans-serif;
    }
    
    .chatbot-toggle {
      width: 48px;
      height: 48px;
      border-radius: 50%;
      background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 100%);
      border: none;
      cursor: pointer;
      box-shadow: 0 4px 20px rgba(224, 71, 60, 0.4);
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.3s ease;
      color: white;
      font-size: 20px;
    }
    
    .chatbot-toggle:hover {
      transform: scale(1.1);
      box-shadow: 0 6px 24px rgba(224, 71, 60, 0.6);
    }
    
    .chatbot-toggle.active {
      transform: rotate(90deg);
    }
    
    .chatbot-window {
      position: absolute;
      bottom: 60px;
      left: 0;
      width: var(--window-width);
      height: var(--window-height);
      background: var(--bg);
      border: 2px solid var(--accent-2);
      border-radius: 12px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
      display: none;
      flex-direction: column;
      overflow: hidden;
      backdrop-filter: blur(10px);
    }
    
    .chatbot-window.active {
      display: flex;
      animation: slideUp 0.3s ease;
    }
    
    @keyframes slideUp {
      from {
        opacity: 0;
        transform: translateY(20px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    
    .chatbot-header {
      position: relative;
      padding: 10px 14px;
      background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 100%);
      color: white;
      font-weight: 600;
      font-size: 13px;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .chatbot-close {
      position: absolute;
      right: 8px;
      top: 8px;
      background: transparent;
      border: none;
      color: rgba(255,255,255,0.95);
      font-size: 16px;
      padding: 6px;
      border-radius: 6px;
      cursor: pointer;
      transition: background 0.15s ease;
    }
    .chatbot-close:hover { background: rgba(255,255,255,0.04); }
    
    .chatbot-status {
      width: 8px;
      height: 8px;
      background: #4ade80;
      border-radius: 50%;
      animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }
    
    .chatbot-messages {
      flex: 1;
      padding: 12px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    
    .chatbot-messages::-webkit-scrollbar {
      width: 6px;
    }
    
    .chatbot-messages::-webkit-scrollbar-track {
      background: rgba(30, 30, 30, 0.3);
    }
    
    .chatbot-messages::-webkit-scrollbar-thumb {
      background: var(--accent-2);
      border-radius: 3px;
    }
    
    .chat-message {
      max-width: 80%;
      padding: 12px 16px;
      border-radius: 12px;
      font-size: 14px;
      line-height: 1.5;
      animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
      from {
        opacity: 0;
        transform: translateY(10px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    
    .chat-message.bot {
      align-self: flex-start;
      background: rgba(224, 71, 60, 0.15);
      border: 1px solid rgba(224, 71, 60, 0.3);
      color: #e6e9ee;
    }
    
    .chat-message.user {
      align-self: flex-end;
      background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 100%);
      color: white;
    }
    
    .chatbot-input-area {
      padding: 12px;
      background: rgba(20, 22, 24, 0.8);
      border-top: 1px solid var(--accent-2);
      display: flex;
      gap: 8px;
    }
    
    .chatbot-input {
      flex: 1;
      background: rgba(30, 32, 34, 0.6);
      border: 1px solid rgba(224, 71, 60, 0.3);
      border-radius: 6px;
      padding: 8px 10px;
      color: #e6e9ee;
      font-size: 13px;
      font-family: Inter, sans-serif;
      outline: none;
      transition: all 0.2s ease;
    }
    
    .chatbot-input:focus {
      border-color: var(--accent-2);
      box-shadow: 0 0 0 3px rgba(224, 71, 60, 0.1);
    }
    
    .chatbot-send {
      background: linear-gradient(135deg, var(--accent-1) 0%, var(--accent-2) 100%);
      border: none;
      border-radius: 8px;
      padding: 8px 12px;
      color: white;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    
    .chatbot-send:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(224, 71, 60, 0.4);
    }
    
    .chatbot-send:active {
      transform: translateY(0);
    }
  `;
  document.head.appendChild(style);
  
  // Create chatbot HTML structure
  const chatbotHTML = `
    <div class="chatbot-container">
      <button class="chatbot-toggle" id="chatbot-toggle" aria-label="Toggle chatbot">
        💬
      </button>
      <div class="chatbot-window" id="chatbot-window">
        <div class="chatbot-header">
          <div class="chatbot-status"></div>
          <span>Neurofoundry Assistant</span>
          <button class="chatbot-close" id="chatbot-close" aria-label="Close">✕</button>
        </div>
        <div class="chatbot-messages" id="chatbot-messages">
          <div class="chat-message bot">
            Hello! I'm your Neurofoundry AI assistant. How can I help you today?
          </div>
        </div>
        <div class="chatbot-input-area">
          <input 
            type="text" 
            class="chatbot-input" 
            id="chatbot-input" 
            placeholder="Type your message..."
            autocomplete="off"
          />
          <button class="chatbot-send" id="chatbot-send">Send</button>
        </div>
      </div>
    </div>
  `;
  
  // Insert chatbot into page when DOM is ready
  function initChatbot() {
    document.body.insertAdjacentHTML('beforeend', chatbotHTML);
    
    const toggle = document.getElementById('chatbot-toggle');
    const window = document.getElementById('chatbot-window');
    const input = document.getElementById('chatbot-input');
    const send = document.getElementById('chatbot-send');
    const messages = document.getElementById('chatbot-messages');
    const closeBtn = document.getElementById('chatbot-close');

    // Close button behavior
    if (closeBtn) {
      closeBtn.addEventListener('click', () => {
        toggle.classList.remove('active');
        window.classList.remove('active');
      });
    }

    // Load config and apply UI settings
    loadExternalConfig().then(cfg => {
      document.documentElement.style.setProperty('--accent-1', cfg.accent1 || defaultConfig.accent1);
      document.documentElement.style.setProperty('--accent-2', cfg.accent2 || defaultConfig.accent2);
      document.documentElement.style.setProperty('--bg', cfg.bg || defaultConfig.bg);
      document.documentElement.style.setProperty('--window-width', (cfg.window_width || defaultConfig.window_width) + 'px');
      document.documentElement.style.setProperty('--window-height', (cfg.window_height || defaultConfig.window_height) + 'px');

      const firstBot = messages.querySelector('.chat-message.bot');
      if (firstBot) firstBot.textContent = cfg.initial_message || defaultConfig.initial_message;
      input.placeholder = cfg.placeholder || defaultConfig.placeholder;

      // ensure chatConfig holds merged values
      Object.assign(chatConfig, cfg);
    }).catch(err => console.warn('Load config failed', err));

    // Close on Escape
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && window.classList.contains('active')) {
        toggle.classList.remove('active');
        window.classList.remove('active');
      }
    });
    
    // Toggle chatbot window
    toggle.addEventListener('click', () => {
      toggle.classList.toggle('active');
      window.classList.toggle('active');
      if (window.classList.contains('active')) {
        input.focus();
      }
    });
    
    // Send message function
    async function sendMessage() {
      const text = input.value.trim();
      if (!text) return;
      
      // Add user message
      const userMsg = document.createElement('div');
      userMsg.className = 'chat-message user';
      userMsg.textContent = text;
      messages.appendChild(userMsg);
      
      input.value = '';
      messages.scrollTop = messages.scrollHeight;
      
      // Show typing indicator
      const typingIndicator = document.createElement('div');
      typingIndicator.className = 'chat-message bot';
      typingIndicator.innerHTML = '<span style="opacity: 0.6;">' + (chatConfig.typing_text || defaultConfig.typing_text) + '</span>';
      typingIndicator.id = 'typing-indicator';
      messages.appendChild(typingIndicator);
      messages.scrollTop = messages.scrollHeight;

      // Personality / rule-based preflight handling
      let personalityHints = [];
      try {
        const rules = (chatConfig.personality && chatConfig.personality.rules) || [];
        const fuzzyEnabled = chatConfig.match_fuzzy === undefined ? defaultConfig.match_fuzzy : chatConfig.match_fuzzy;
        const fuzzyThreshold = chatConfig.fuzzy_threshold === undefined ? defaultConfig.fuzzy_threshold : chatConfig.fuzzy_threshold;
        const fuzzyWhole = chatConfig.fuzzy_whole_word === undefined ? defaultConfig.fuzzy_whole_word : chatConfig.fuzzy_whole_word;

        for (const rule of rules) {
          try {
            let matched = false;

            // 1) Regex match if provided
            if (rule.match_regex) {
              const re = new RegExp(rule.match_regex, 'i');
              matched = re.test(text);
            }

            // 2) Exact / substring match (case-insensitive)
            if (!matched && rule.match) {
              const needle = String(rule.match).toLowerCase();
              matched = text.toLowerCase().includes(needle);
            }

            // 3) Fuzzy match fallback
            if (!matched && fuzzyEnabled && rule.match) {
              const target = String(rule.match).toLowerCase().trim();
              if (fuzzyWhole) {
                // split input into tokens and compare tokens to target
                const tokens = text.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
                for (const tkn of tokens) {
                  const sim = similarity(tkn, target);
                  if (sim >= fuzzyThreshold) { matched = true; break; }
                }
              } else {
                const sim = similarity(text.toLowerCase(), target);
                if (sim >= fuzzyThreshold) matched = true;
              }
            }

            if (!matched) continue;

            // Resolve response text (direct string or response_set)
            let responseText = null;
            if (rule.response) {
              if (typeof rule.response === 'string') responseText = rule.response;
              else if (rule.response.response_set) {
                const set = (chatConfig.response_sets && chatConfig.response_sets[rule.response.response_set]) || [];
                if (Array.isArray(set) && set.length) responseText = set[Math.floor(Math.random() * set.length)];
              }
            }

            // If configured to reply locally, show and optionally stop propagation
            const action = rule.action || 'reply';
            const stop = rule.stop_propagation !== false; // default true
            if (action === 'reply' && responseText) {
              const botMsg = document.createElement('div');
              botMsg.className = 'chat-message bot';
              botMsg.textContent = responseText;
              messages.appendChild(botMsg);

              // Remove typing indicator and skip network if stop is true
              const indicator = document.getElementById('typing-indicator');
              if (indicator) indicator.remove();
              if (stop) { messages.scrollTop = messages.scrollHeight; return; }
            }

            // If action is inject, add to personalityHints to send to endpoint
            if (action === 'inject') {
              if (responseText) personalityHints.push(responseText);
              else personalityHints.push(rule.match || rule.name || '');
            }

            // continue to next rule
          } catch (e) { console.warn('Rule evaluation error', e); }
        }
      } catch (e) { console.warn('Personality rule processing failed', e); }

      // Create a tracing span for the chat request (if tracer available)
      const tracer = window.otelTracer;
      const chatSpan = tracer ? tracer.startSpan('chat.request', { attributes: { component: 'chatbot', method: 'POST' } }) : null;
      try {
        // Call configured endpoint (include assistant instructions and CTA)
        const payload = {
          prompt: text,
          conversation_history: [],  // Can add conversation tracking later
          personality_hints: personalityHints,
          personality: chatConfig.personality && chatConfig.personality.name,
          assistant_instructions: chatConfig.assistant_instructions || defaultConfig.assistant_instructions,
          cta: { text: chatConfig.cta_text || defaultConfig.cta_text, url: chatConfig.cta_url || defaultConfig.cta_url },
          response_policy: { max_length: chatConfig.max_response_length || defaultConfig.max_response_length, shorten: chatConfig.shorten_policy || defaultConfig.shorten_policy }
        };

        const response = await fetch(chatConfig.endpoint || defaultConfig.endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        if (chatSpan) {
          chatSpan.setAttribute('http.status_code', response.status);
          chatSpan.end();
        }
        
        // Remove typing indicator
        const indicator = document.getElementById('typing-indicator');
        if (indicator) indicator.remove();
        
        if (data.ok && data.content) {
          // Post-process content according to response policy to avoid encyclopedic replies
          function postProcessResponse(text) {
            const maxLen = chatConfig.max_response_length || defaultConfig.max_response_length || 300;
            const shorten = (chatConfig.shorten_policy !== undefined) ? chatConfig.shorten_policy : defaultConfig.shorten_policy;
            const ctaText = (chatConfig.cta_text || defaultConfig.cta_text) || '';
            const ctaUrl = (chatConfig.cta_url || defaultConfig.cta_url) || '';
            const cta = ctaText ? ('\n\n' + ctaText + ' ' + ctaUrl) : '';

            // crude encyclopedic detection
            const epicRegex = /\b(Wikipedia|According to|In \d{4}|References|Source:|Citation:|For example)\b/i;
            if ((shorten && text.length > maxLen) || epicRegex.test(text)) {
              // take the first reasonable chunk (up to maxLen) and append CTA
              const short = text.slice(0, maxLen).trim();
              return short + '...' + cta;
            }
            return text;
          }

          const processed = postProcessResponse(data.content);
          const botMsg = document.createElement('div');
          botMsg.className = 'chat-message bot';
          botMsg.textContent = processed;
          messages.appendChild(botMsg);
        } else {
          throw new Error(data.detail || 'Failed to get response');
        }
        
      } catch (error) {
        console.error('Chat error:', error);
        
        // Remove typing indicator
        const indicator = document.getElementById('typing-indicator');
        if (indicator) indicator.remove();
        
        // Mark error on span (if present) and end it
        if (chatSpan) {
          try { chatSpan.setAttribute('error', true); } catch(e){}
          try { chatSpan.end(); } catch(e){}
        }

        // Show error message
        const botMsg = document.createElement('div');
        botMsg.className = 'chat-message bot';
        botMsg.innerHTML = '<span style="opacity: 0.6;">' + (chatConfig.error_message || defaultConfig.error_message) + '</span>';
        messages.appendChild(botMsg);
      }
      
      messages.scrollTop = messages.scrollHeight;
    }
    
    // Send on button click
    send.addEventListener('click', sendMessage);
    
    // Send on Enter key
    input.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        sendMessage();
      }
    });
  }
  
  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initChatbot);
  } else {
    initChatbot();
  }
})();
