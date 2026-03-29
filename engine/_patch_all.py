"""Patch app.py with all 5 bug fixes."""
import re

with open('app.py', 'rb') as f:
    raw = f.read()
if raw[:3] == b'\xef\xbb\xbf':
    raw = raw[3:]
content = raw.decode('utf-8')
original = content  # save for verification

# ═══════════════════════════════════════════════════════════════
# FIX 1: Execute Scan Sound — play every time the button is hit
# ═══════════════════════════════════════════════════════════════
SCAN_SOUND = '''
    # ── Scan activation sound ──
    import streamlit.components.v1 as _scan_snd
    _scan_snd.html("""<script>
try {
  const W = window.parent || window;
  const A = new (W.AudioContext || W.webkitAudioContext)();
  A.resume();
  [523,659,784,1047].forEach((f,i) => {
    const o = A.createOscillator(); o.type='sine'; o.frequency.value=f;
    const g = A.createGain();
    g.gain.setValueAtTime(0, A.currentTime + i*0.06);
    g.gain.linearRampToValueAtTime(0.08, A.currentTime + i*0.06 + 0.02);
    g.gain.exponentialRampToValueAtTime(0.001, A.currentTime + i*0.06 + 0.2);
    o.connect(g); g.connect(A.destination);
    o.start(A.currentTime + i*0.06); o.stop(A.currentTime + i*0.06 + 0.25);
  });
} catch(e){}
</script>""", height=0)'''

old1 = "    result['asset'] = asset\n    st.session_state.data = result"
new1 = "    result['asset'] = asset\n    st.session_state.data = result\n" + SCAN_SOUND
assert old1 in content, "FIX1: anchor not found"
content = content.replace(old1, new1, 1)
print("FIX 1 OK: scan sound")

# ═══════════════════════════════════════════════════════════════
# FIX 2: Shutdown terminal — fullscreen overlay so it's visible
#         regardless of scroll position
# ═══════════════════════════════════════════════════════════════
old2 = '''if st.session_state.entered == 'shutdown':
    st.markdown("""<style>
[data-testid="stSidebar"]{display:none;}
.block-container{padding:0!important;max-width:100%!important;}
[data-testid="stAppViewContainer"]>div:first-child{padding-top:0;}
header{display:none!important;}
iframe{border:none!important;}
</style>""", unsafe_allow_html=True)'''

new2 = '''if st.session_state.entered == 'shutdown':
    st.markdown("""<style>
[data-testid="stSidebar"]{display:none;}
.block-container{padding:0!important;max-width:100%!important;}
[data-testid="stAppViewContainer"]>div:first-child{padding-top:0;}
header{display:none!important;}
iframe{border:none!important;}
/* Fullscreen shutdown overlay */
[data-testid="stAppViewContainer"] iframe[height="700"]{
    position:fixed!important;top:0!important;left:0!important;
    width:100vw!important;height:100vh!important;z-index:99999!important;
    border:none!important;
}
</style>""", unsafe_allow_html=True)
    # Scroll to top so shutdown is visible
    import streamlit.components.v1 as _sd_scroll
    _sd_scroll.html("<script>window.parent.document.querySelector('[data-testid=\\\\'stAppViewContainer\\\\']').scrollTo(0,0);</script>",height=0)'''

assert old2 in content, "FIX2: anchor not found"
content = content.replace(old2, new2, 1)
print("FIX 2 OK: shutdown fullscreen")

# ═══════════════════════════════════════════════════════════════
# FIX 3: Home page — animated typing on keyboard keys
# ═══════════════════════════════════════════════════════════════
# Add typing keyframes to the home page CSS
old3 = "@keyframes cursorBlink {"
new3 = """@keyframes keyType {
  0%,80%,100% { opacity:0.1; }
  85%,95% { opacity:0.9; fill:rgba(0,229,255,0.7); filter:drop-shadow(0 0 3px rgba(0,229,255,0.5)); }
}
@keyframes cursorBlink {"""
assert old3 in content, "FIX3a: anchor not found"
content = content.replace(old3, new3, 1)

# Now add animation classes to keyboard key rects — they all have y='112'
# Replace static opacity keyboard keys with animated ones
kbd_keys_old = """    <rect x='-20' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.15'/>
    <rect x='-12' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1'/>
    <rect x='-4' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.2'/>
    <rect x='4' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1'/>
    <rect x='20' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.15'/>
    <rect x='28' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1'/>
    <rect x='50' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.2'/>
    <rect x='70' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1'/>
    <rect x='90' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.15'/>
    <rect x='100' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1'/>
    <rect x='110' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.2'/>"""

# Animated keys with staggered delays simulating typing
kbd_keys_new = """    <rect x='-20' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 0.0s'/>
    <rect x='-12' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 0.35s'/>
    <rect x='-4' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 0.15s'/>
    <rect x='4' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 0.55s'/>
    <rect x='20' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 0.8s'/>
    <rect x='28' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 0.25s'/>
    <rect x='50' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 1.1s'/>
    <rect x='70' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 0.45s'/>
    <rect x='90' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 0.7s'/>
    <rect x='100' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 1.3s'/>
    <rect x='110' y='112' width='4' height='3' rx='1' fill='{CYAN}' opacity='0.1' style='animation:keyType 1.8s ease infinite 0.95s'/>"""

# Also add a blinking cursor on the center monitor
cursor_line = """    <text x='360' y='108' fill='{CYAN}' font-family='JetBrains Mono,monospace' font-size='9.5' font-weight='700'>&#9670; NEXUS FUSION</text>"""
cursor_new = """    <text x='360' y='108' fill='{CYAN}' font-family='JetBrains Mono,monospace' font-size='9.5' font-weight='700'>&#9670; NEXUS FUSION</text>
    <rect x='538' y='132' width='6' height='10' rx='1' fill='{CYAN}' opacity='0.7' style='animation:cursorBlink 1s step-end infinite'/>"""

assert kbd_keys_old in content, "FIX3b: keyboard anchor not found"
content = content.replace(kbd_keys_old, kbd_keys_new, 1)
assert cursor_line in content, "FIX3c: cursor anchor not found"
content = content.replace(cursor_line, cursor_new, 1)
print("FIX 3 OK: typing animation")

# ═══════════════════════════════════════════════════════════════
# FIX 4 & 5: Oracle AI as floating side popup with auto-scroll
# ═══════════════════════════════════════════════════════════════

# 4a. Add floating Oracle CSS to the main CSS block
# Find the oracle CSS section and enhance it
old_oracle_css = """.oracle-chat-container {{"""
new_oracle_css = """.oracle-float-panel {{
    position: fixed !important;
    bottom: 20px;
    right: 20px;
    width: 400px;
    max-height: 520px;
    z-index: 99999;
    background: rgba(8,12,24,0.98);
    border: 1px solid rgba(179,136,255,0.35);
    border-radius: 14px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.6), 0 0 30px rgba(179,136,255,0.1);
    backdrop-filter: blur(20px);
    overflow: hidden;
    display: flex;
    flex-direction: column;
}}
.oracle-float-panel .oracle-float-header {{
    padding: 14px 16px;
    background: linear-gradient(135deg, rgba(179,136,255,0.12), rgba(0,229,255,0.06));
    border-bottom: 1px solid rgba(179,136,255,0.2);
    display: flex;
    align-items: center;
    gap: 10px;
    flex-shrink: 0;
    cursor: default;
}}
.oracle-float-panel .oracle-float-messages {{
    flex: 1;
    overflow-y: auto;
    padding: 12px 14px;
    max-height: 320px;
    scroll-behavior: smooth;
}}
.oracle-float-panel .oracle-float-messages::-webkit-scrollbar {{ width:4px; }}
.oracle-float-panel .oracle-float-messages::-webkit-scrollbar-track {{ background:transparent; }}
.oracle-float-panel .oracle-float-messages::-webkit-scrollbar-thumb {{ background:rgba(179,136,255,0.3); border-radius:4px; }}
.oracle-chat-container {{"""

assert old_oracle_css in content, "FIX4a: oracle css anchor not found"
content = content.replace(old_oracle_css, new_oracle_css, 1)
print("FIX 4a OK: floating CSS")

# 4b. Replace Oracle in awaiting state with floating version
old_oracle_await = """    # ── Oracle AI in Awaiting State ──────────────────────────────────────────
    if st.session_state.oracle_active:
        with st.container():
            st.markdown(f\"\"\"
<div class='nx-panel' style='border:1px solid rgba(179,136,255,0.3);margin-top:20px;'>
<div style='display:flex;align-items:center;gap:12px;margin-bottom:16px;'>
    <div style='width:42px;height:42px;background:linear-gradient(135deg,{PURPLE},{CYAN});border-radius:50%;
    display:flex;align-items:center;justify-content:center;font-size:1.3rem;box-shadow:0 0 20px rgba(179,136,255,0.5);'>◇</div>
    <div>
        <div style='color:{PURPLE};font-size:1.15rem;font-weight:700;letter-spacing:2px;font-family:Inter,sans-serif;'>ORACLE AI</div>
        <div style='color:{DIM};font-size:0.8rem;'>RAG-Powered Market Intelligence</div>
    </div>
    <div style='margin-left:auto;display:flex;align-items:center;gap:6px;'>
        <div style='width:8px;height:8px;background:{GREEN};border-radius:50%;box-shadow:0 0 8px {GREEN};'></div>
        <span style='color:{GREEN};font-size:0.8rem;'>ONLINE</span>
    </div>
</div>
</div>\"\"\", unsafe_allow_html=True)
            
            chat_container = st.container(height=300)
            with chat_container:
                if not st.session_state.oracle_messages:
                    st.markdown(f\"\"\"
<div style='text-align:center;padding:30px 20px;'>
    <div style='font-size:2rem;margin-bottom:10px;'>◇</div>
    <div style='color:{PURPLE};font-size:1rem;font-weight:600;margin-bottom:6px;'>Oracle AI Ready</div>
    <div style='color:{DIM};font-size:0.85rem;'>Ask about market data or run a scan first.</div>
</div>\"\"\", unsafe_allow_html=True)
                else:
                    for msg in st.session_state.oracle_messages:
                        if msg['role'] == 'user':
                            st.markdown(f\"\"\"
<div style='display:flex;justify-content:flex-end;margin:6px 0;'>
    <div style='background:linear-gradient(135deg,rgba(0,229,255,0.15),rgba(0,229,255,0.08));
    border:1px solid rgba(0,229,255,0.3);padding:10px 14px;border-radius:10px 10px 2px 10px;
    max-width:80%;color:#eef2f7;font-size:0.9rem;'>{msg['content']}</div>
</div>\"\"\", unsafe_allow_html=True)
                        else:
                            st.markdown(f\"\"\"
<div style='display:flex;justify-content:flex-start;margin:6px 0;gap:8px;'>
    <div style='width:24px;height:24px;background:linear-gradient(135deg,{PURPLE},{CYAN});border-radius:50%;
    display:flex;align-items:center;justify-content:center;font-size:0.7rem;flex-shrink:0;'>◇</div>
    <div style='background:linear-gradient(135deg,rgba(179,136,255,0.12),rgba(179,136,255,0.06));
    border:1px solid rgba(179,136,255,0.25);padding:10px 14px;border-radius:2px 10px 10px 10px;
    max-width:80%;color:#c0c8d8;font-size:0.9rem;line-height:1.4;'>{msg['content']}</div>
</div>\"\"\", unsafe_allow_html=True)
            
            user_q = st.chat_input("Ask Oracle about market data...", key="oracle_chat_await")
            
            if user_q:
                st.session_state.oracle_messages.append({"role": "user", "content": user_q})
                try:
                    response = _ask_oracle(st.session_state.oracle_messages, st.session_state.data)
                    st.session_state.oracle_messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.session_state.oracle_messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
                st.rerun()
            
            if st.session_state.oracle_messages:
                if st.button("Clear", key="clear_await"):
                    st.session_state.oracle_messages = []
                    st.rerun()"""

# Build the floating Oracle chat function that we'll use in both locations
FLOAT_ORACLE_FUNC = '''
# ═══════════════════════════════════════════════════════════════════════════════
# Oracle AI — Floating Side Panel (shared renderer)
# ═══════════════════════════════════════════════════════════════════════════════
def _render_oracle_panel(key_suffix="main"):
    """Render the floating Oracle AI chat panel."""
    import streamlit.components.v1 as _oracle_comp

    # Build messages HTML
    msgs_html = ""
    if not st.session_state.oracle_messages:
        msgs_html = f"""
        <div style='text-align:center;padding:30px 16px;'>
            <div style='font-size:2rem;margin-bottom:8px;'>◇</div>
            <div style='color:{PURPLE};font-size:0.95rem;font-weight:600;margin-bottom:6px;'>Oracle AI Ready</div>
            <div style='color:{DIM};font-size:0.82rem;'>Ask about your analysis, signals, or market data.</div>
        </div>"""
    else:
        for msg in st.session_state.oracle_messages:
            if msg['role'] == 'user':
                msgs_html += f"""
                <div style='display:flex;justify-content:flex-end;margin:8px 0;'>
                    <div style='background:linear-gradient(135deg,rgba(0,229,255,0.15),rgba(0,229,255,0.08));
                    border:1px solid rgba(0,229,255,0.3);padding:10px 14px;border-radius:10px 10px 2px 10px;
                    max-width:85%;color:#eef2f7;font-size:0.88rem;word-wrap:break-word;'>{msg['content']}</div>
                </div>"""
            else:
                msgs_html += f"""
                <div style='display:flex;justify-content:flex-start;margin:8px 0;gap:8px;'>
                    <div style='width:22px;height:22px;min-width:22px;background:linear-gradient(135deg,{PURPLE},{CYAN});border-radius:50%;
                    display:flex;align-items:center;justify-content:center;font-size:0.65rem;flex-shrink:0;margin-top:2px;'>◇</div>
                    <div style='background:linear-gradient(135deg,rgba(179,136,255,0.12),rgba(179,136,255,0.06));
                    border:1px solid rgba(179,136,255,0.25);padding:10px 14px;border-radius:2px 10px 10px 10px;
                    max-width:85%;color:#c0c8d8;font-size:0.88rem;line-height:1.5;word-wrap:break-word;'>{msg['content']}</div>
                </div>"""

    # Render floating panel via component (pure HTML = truly fixed position)
    panel_html = f"""<div class="oracle-float-panel">
        <div class="oracle-float-header">
            <div style='width:32px;height:32px;background:linear-gradient(135deg,{PURPLE},{CYAN});border-radius:50%;
            display:flex;align-items:center;justify-content:center;font-size:1rem;
            box-shadow:0 0 15px rgba(179,136,255,0.4);flex-shrink:0;'>◇</div>
            <div style='flex:1;'>
                <div style='color:{PURPLE};font-size:0.95rem;font-weight:700;letter-spacing:2px;font-family:Inter,sans-serif;'>ORACLE AI</div>
                <div style='color:{DIM};font-size:0.72rem;'>Groq-Powered Intelligence</div>
            </div>
            <div style='display:flex;align-items:center;gap:5px;'>
                <div style='width:7px;height:7px;background:{GREEN};border-radius:50%;box-shadow:0 0 6px {GREEN};'></div>
                <span style='color:{GREEN};font-size:0.72rem;'>LIVE</span>
            </div>
        </div>
        <div class="oracle-float-messages" id="oracle-msgs">
            {{msgs_html}}
        </div>
    </div>
    <script>
    // Auto-scroll to bottom
    const mc = document.getElementById('oracle-msgs');
    if (mc) mc.scrollTop = mc.scrollHeight;
    </script>"""

    _oracle_comp.html(f"""<!DOCTYPE html><html><head>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
    body {{ margin:0; padding:0; background:transparent; overflow:visible; font-family:Inter,sans-serif; }}
    .oracle-float-panel {{
        position:fixed; bottom:20px; right:20px; width:390px; max-height:500px;
        background:rgba(8,12,24,0.98); border:1px solid rgba(179,136,255,0.35);
        border-radius:14px; box-shadow:0 8px 40px rgba(0,0,0,0.6),0 0 30px rgba(179,136,255,0.1);
        backdrop-filter:blur(20px); overflow:hidden; display:flex; flex-direction:column;
        z-index:99999;
    }}
    .oracle-float-header {{
        padding:12px 14px; background:linear-gradient(135deg,rgba(179,136,255,0.12),rgba(0,229,255,0.06));
        border-bottom:1px solid rgba(179,136,255,0.2); display:flex; align-items:center; gap:10px;
    }}
    .oracle-float-messages {{
        flex:1; overflow-y:auto; padding:10px 12px; max-height:320px; scroll-behavior:smooth;
    }}
    .oracle-float-messages::-webkit-scrollbar {{ width:4px; }}
    .oracle-float-messages::-webkit-scrollbar-thumb {{ background:rgba(179,136,255,0.3); border-radius:4px; }}
    .oracle-float-messages::-webkit-scrollbar-track {{ background:transparent; }}
    </style></head><body>{{panel_html}}</body></html>""".replace("{{msgs_html}}", msgs_html).replace("{{panel_html}}", panel_html.replace("{msgs_html}", msgs_html)), height=0)

    # Streamlit chat input + processing (outside the HTML component, in normal flow)
    user_q = st.chat_input(f"Ask Oracle anything...", key=f"oracle_chat_{{key_suffix}}")
    if user_q:
        st.session_state.oracle_messages.append({{"role": "user", "content": user_q}})
        try:
            response = _ask_oracle(st.session_state.oracle_messages, st.session_state.data)
            st.session_state.oracle_messages.append({{"role": "assistant", "content": response}})
        except Exception as e:
            st.session_state.oracle_messages.append({{"role": "assistant", "content": f"Error: {{str(e)}}"}})
        st.rerun()

    col1, col2 = st.columns([6, 1])
    with col2:
        if st.session_state.oracle_messages:
            if st.button("Clear", key=f"clear_{{key_suffix}}"):
                st.session_state.oracle_messages = []
                st.rerun()'''

# Hmm, this approach of building the function string is getting messy with all the escaping. Let me take a simpler approach.
# I'll keep the Streamlit widgets as-is but render the message history in a fixed-position HTML component, 
# and leave the chat_input in the main flow (it pins to bottom by default).

# SIMPLER APPROACH: Just restyle the existing Oracle sections to float using CSS injection.
# The trick: wrap the oracle in a container, inject CSS to make it fixed.

new_oracle_await = """    # ── Oracle AI in Awaiting State — Floating Side Panel ─────────────────────
    if st.session_state.oracle_active:
        # Inject CSS to float the oracle panel
        st.markdown(f\"\"\"<style>
div[data-testid="stVerticalBlock"]:has(> div.oracle-await-marker) {{
    position:fixed !important; bottom:20px; right:20px; width:400px; z-index:99999;
    background:rgba(8,12,24,0.98); border:1px solid rgba(179,136,255,0.35);
    border-radius:14px; box-shadow:0 8px 40px rgba(0,0,0,0.6),0 0 30px rgba(179,136,255,0.1);
    backdrop-filter:blur(20px); padding:0 12px 12px; max-height:520px; overflow:visible;
}}
div[data-testid="stVerticalBlock"]:has(> div.oracle-await-marker) [data-testid="stChatInput"] {{
    position:sticky; bottom:0; background:rgba(8,12,24,0.98);
}}
</style>\"\"\", unsafe_allow_html=True)
        with st.container():
            st.markdown("<div class='oracle-await-marker' style='display:none'></div>", unsafe_allow_html=True)
            st.markdown(f\"\"\"
<div style='padding:12px 4px 8px;display:flex;align-items:center;gap:10px;
border-bottom:1px solid rgba(179,136,255,0.2);margin:0 -12px;padding-left:16px;padding-right:16px;
background:linear-gradient(135deg,rgba(179,136,255,0.1),rgba(0,229,255,0.04));'>
    <div style='width:32px;height:32px;background:linear-gradient(135deg,{PURPLE},{CYAN});border-radius:50%;
    display:flex;align-items:center;justify-content:center;font-size:1rem;
    box-shadow:0 0 15px rgba(179,136,255,0.4);flex-shrink:0;'>◇</div>
    <div style='flex:1;'>
        <div style='color:{PURPLE};font-size:0.95rem;font-weight:700;letter-spacing:2px;font-family:Inter,sans-serif;'>ORACLE AI</div>
        <div style='color:{DIM};font-size:0.72rem;'>Groq-Powered Intelligence</div>
    </div>
    <div style='display:flex;align-items:center;gap:5px;'>
        <div style='width:7px;height:7px;background:{GREEN};border-radius:50%;box-shadow:0 0 6px {GREEN};'></div>
        <span style='color:{GREEN};font-size:0.72rem;'>LIVE</span>
    </div>
</div>\"\"\", unsafe_allow_html=True)

            chat_container = st.container(height=280)
            with chat_container:
                if not st.session_state.oracle_messages:
                    st.markdown(f\"\"\"
<div style='text-align:center;padding:30px 16px;'>
    <div style='font-size:2rem;margin-bottom:8px;'>◇</div>
    <div style='color:{PURPLE};font-size:0.95rem;font-weight:600;margin-bottom:6px;'>Oracle AI Ready</div>
    <div style='color:{DIM};font-size:0.82rem;'>Ask about market data or run a scan first.</div>
</div>\"\"\", unsafe_allow_html=True)
                else:
                    for msg in st.session_state.oracle_messages:
                        if msg['role'] == 'user':
                            st.markdown(f\"\"\"
<div style='display:flex;justify-content:flex-end;margin:6px 0;'>
    <div style='background:linear-gradient(135deg,rgba(0,229,255,0.15),rgba(0,229,255,0.08));
    border:1px solid rgba(0,229,255,0.3);padding:10px 14px;border-radius:10px 10px 2px 10px;
    max-width:85%;color:#eef2f7;font-size:0.88rem;word-wrap:break-word;'>{msg['content']}</div>
</div>\"\"\", unsafe_allow_html=True)
                        else:
                            st.markdown(f\"\"\"
<div style='display:flex;justify-content:flex-start;margin:6px 0;gap:8px;'>
    <div style='width:22px;height:22px;min-width:22px;background:linear-gradient(135deg,{PURPLE},{CYAN});border-radius:50%;
    display:flex;align-items:center;justify-content:center;font-size:0.65rem;flex-shrink:0;margin-top:2px;'>◇</div>
    <div style='background:linear-gradient(135deg,rgba(179,136,255,0.12),rgba(179,136,255,0.06));
    border:1px solid rgba(179,136,255,0.25);padding:10px 14px;border-radius:2px 10px 10px 10px;
    max-width:85%;color:#c0c8d8;font-size:0.88rem;line-height:1.5;word-wrap:break-word;'>{msg['content']}</div>
</div>\"\"\", unsafe_allow_html=True)

            # Auto-scroll chat to bottom
            import streamlit.components.v1 as _oscroll
            _oscroll.html(\"\"\"<script>
try {
  const f = window.parent.document;
  const containers = f.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]');
  containers.forEach(c => {
    const inner = c.querySelector('[data-testid="stScrollableContainer"]');
    if (inner) inner.scrollTop = inner.scrollHeight;
  });
} catch(e){}
</script>\"\"\", height=0)

            user_q = st.chat_input("Ask Oracle anything...", key="oracle_chat_await")
            if user_q:
                st.session_state.oracle_messages.append({"role": "user", "content": user_q})
                try:
                    response = _ask_oracle(st.session_state.oracle_messages, st.session_state.data)
                    st.session_state.oracle_messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.session_state.oracle_messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
                st.rerun()

            if st.session_state.oracle_messages:
                if st.button("Clear", key="clear_await"):
                    st.session_state.oracle_messages = []
                    st.rerun()"""

assert old_oracle_await in content, "FIX4b: awaiting oracle anchor not found"
content = content.replace(old_oracle_await, new_oracle_await, 1)
print("FIX 4b OK: awaiting oracle floating")

# 4c. Replace Oracle in scan results (bottom) with floating version
# Find the entire scan results Oracle section
old_oracle_scan_marker = """# ═══════════════════════════════════════════════════════════════════════════════
# ORACLE AI — Conversational Interface (Floating Panel)
# ═══════════════════════════════════════════════════════════════════════════════"""
assert old_oracle_scan_marker in content, "FIX4c: scan oracle marker not found"

# Find everything from the marker to end of file and replace
idx = content.index(old_oracle_scan_marker)
old_oracle_scan = content[idx:]

new_oracle_scan = """# ═══════════════════════════════════════════════════════════════════════════════
# ORACLE AI — Floating Side Panel (scan results)
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.oracle_active:
    # Inject CSS to float the oracle panel
    st.markdown(f\"\"\"<style>
div[data-testid="stVerticalBlock"]:has(> div.oracle-scan-marker) {{
    position:fixed !important; bottom:20px; right:20px; width:400px; z-index:99999;
    background:rgba(8,12,24,0.98); border:1px solid rgba(179,136,255,0.35);
    border-radius:14px; box-shadow:0 8px 40px rgba(0,0,0,0.6),0 0 30px rgba(179,136,255,0.1);
    backdrop-filter:blur(20px); padding:0 12px 12px; max-height:520px; overflow:visible;
}}
div[data-testid="stVerticalBlock"]:has(> div.oracle-scan-marker) [data-testid="stChatInput"] {{
    position:sticky; bottom:0; background:rgba(8,12,24,0.98);
}}
</style>\"\"\", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='oracle-scan-marker' style='display:none'></div>", unsafe_allow_html=True)
        st.markdown(f\"\"\"
<div style='padding:12px 4px 8px;display:flex;align-items:center;gap:10px;
border-bottom:1px solid rgba(179,136,255,0.2);margin:0 -12px;padding-left:16px;padding-right:16px;
background:linear-gradient(135deg,rgba(179,136,255,0.1),rgba(0,229,255,0.04));'>
    <div style='width:32px;height:32px;background:linear-gradient(135deg,{PURPLE},{CYAN});border-radius:50%;
    display:flex;align-items:center;justify-content:center;font-size:1rem;
    box-shadow:0 0 15px rgba(179,136,255,0.4);flex-shrink:0;'>◇</div>
    <div style='flex:1;'>
        <div style='color:{PURPLE};font-size:0.95rem;font-weight:700;letter-spacing:2px;font-family:Inter,sans-serif;'>ORACLE AI</div>
        <div style='color:{DIM};font-size:0.72rem;'>Groq-Powered Intelligence</div>
    </div>
    <div style='display:flex;align-items:center;gap:5px;'>
        <div style='width:7px;height:7px;background:{GREEN};border-radius:50%;box-shadow:0 0 6px {GREEN};'></div>
        <span style='color:{GREEN};font-size:0.72rem;'>LIVE</span>
    </div>
</div>\"\"\", unsafe_allow_html=True)

        chat_container = st.container(height=280)
        with chat_container:
            if not st.session_state.oracle_messages:
                st.markdown(f\"\"\"
<div style='text-align:center;padding:30px 16px;'>
    <div style='font-size:2rem;margin-bottom:8px;'>◇</div>
    <div style='color:{PURPLE};font-size:0.95rem;font-weight:600;margin-bottom:6px;'>Oracle AI Ready</div>
    <div style='color:{DIM};font-size:0.82rem;'>Ask about your analysis, signals, or data.</div>
    <div style='margin-top:14px;display:flex;flex-wrap:wrap;gap:6px;justify-content:center;'>
        <span style='background:rgba(179,136,255,0.1);border:1px solid rgba(179,136,255,0.3);
        padding:5px 10px;border-radius:16px;font-size:0.75rem;color:{PURPLE};'>"Explain my signal"</span>
        <span style='background:rgba(0,229,255,0.1);border:1px solid rgba(0,229,255,0.3);
        padding:5px 10px;border-radius:16px;font-size:0.75rem;color:{CYAN};'>"Why is caution high?"</span>
        <span style='background:rgba(0,255,170,0.1);border:1px solid rgba(0,255,170,0.3);
        padding:5px 10px;border-radius:16px;font-size:0.75rem;color:{GREEN};'>"Polymarket outlook?"</span>
    </div>
</div>\"\"\", unsafe_allow_html=True)
            else:
                for msg in st.session_state.oracle_messages:
                    if msg['role'] == 'user':
                        st.markdown(f\"\"\"
<div style='display:flex;justify-content:flex-end;margin:6px 0;'>
    <div style='background:linear-gradient(135deg,rgba(0,229,255,0.15),rgba(0,229,255,0.08));
    border:1px solid rgba(0,229,255,0.3);padding:10px 14px;border-radius:10px 10px 2px 10px;
    max-width:85%;color:#eef2f7;font-size:0.88rem;word-wrap:break-word;'>{msg['content']}</div>
</div>\"\"\", unsafe_allow_html=True)
                    else:
                        st.markdown(f\"\"\"
<div style='display:flex;justify-content:flex-start;margin:6px 0;gap:8px;'>
    <div style='width:22px;height:22px;min-width:22px;background:linear-gradient(135deg,{PURPLE},{CYAN});border-radius:50%;
    display:flex;align-items:center;justify-content:center;font-size:0.65rem;flex-shrink:0;margin-top:2px;'>◇</div>
    <div style='background:linear-gradient(135deg,rgba(179,136,255,0.12),rgba(179,136,255,0.06));
    border:1px solid rgba(179,136,255,0.25);padding:10px 14px;border-radius:2px 10px 10px 10px;
    max-width:85%;color:#c0c8d8;font-size:0.88rem;line-height:1.5;word-wrap:break-word;'>{msg['content']}</div>
</div>\"\"\", unsafe_allow_html=True)

        # Auto-scroll chat to bottom
        import streamlit.components.v1 as _oscroll2
        _oscroll2.html(\"\"\"<script>
try {
  const f = window.parent.document;
  const containers = f.querySelectorAll('[data-testid="stScrollableContainer"]');
  containers.forEach(c => { c.scrollTop = c.scrollHeight; });
} catch(e){}
</script>\"\"\", height=0)

        user_query = st.chat_input("Ask Oracle about your analysis, signals, or market data...", key="oracle_chat_main")
        if user_query:
            st.session_state.oracle_messages.append({"role": "user", "content": user_query})
            try:
                response = _ask_oracle(st.session_state.oracle_messages, st.session_state.data)
                st.session_state.oracle_messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.session_state.oracle_messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
            st.rerun()

        if st.session_state.oracle_messages:
            if st.button("Clear Chat", key="clear_oracle"):
                st.session_state.oracle_messages = []
                st.rerun()
"""

content = content[:idx] + new_oracle_scan
print("FIX 4c OK: scan results oracle floating")

# ═══════════════════════════════════════════════════════════════
# Verify and write
# ═══════════════════════════════════════════════════════════════
assert content != original, "No changes were made!"

# Verify Python syntax
import ast
try:
    ast.parse(content)
    print("Syntax check: OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
    print("Aborting — file not written.")
    import sys; sys.exit(1)

with open('app.py', 'w', encoding='utf-8', newline='') as f:
    f.write(content)

print("\nAll 5 fixes applied successfully!")
print("1. Execute scan sound plays every time")
print("2. Shutdown terminal overlays fullscreen")
print("3. Trader keyboard typing animation")
print("4. Oracle AI is a floating side popup")
print("5. Chat auto-scrolls + Enter works")
